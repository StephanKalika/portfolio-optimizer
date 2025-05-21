from flask import Flask, jsonify, request
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib # For saving the scaler
import json # For saving model config

app = Flask(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
MODELS_DIR = os.path.join(app.root_path, 'trained_models') # Save models inside the app directory for now
os.makedirs(MODELS_DIR, exist_ok=True)

db_engine = None

def initialize_db_engine(retries=5, delay=5):
    global db_engine
    if not DATABASE_URL:
        app.logger.error("DATABASE_URL environment variable is not set.")
        return
    for attempt in range(retries):
        try:
            current_engine = create_engine(DATABASE_URL)
            with current_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            db_engine = current_engine
            app.logger.info(f"Database engine connected after {attempt + 1} attempts.")
            return
        except SQLAlchemyError as e:
            app.logger.error(f"Error creating database engine (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(delay)
            else: db_engine = None

initialize_db_engine()

# --- PyTorch Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[:, -1]) # Get output from the last time step
        return predictions

# --- Data Preprocessing Function ---
def create_sequences_and_scale(data_series, sequence_length=60, test_size=0.2):
    if len(data_series) <= sequence_length:
        raise ValueError("Not enough data to create sequences.")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)

    if test_size > 0 and len(X) > 1: # Ensure there's enough data to split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        # Ensure test set is not empty after split
        if len(X_test) == 0:
             X_train = X # Use all data for training if test split results in empty test set
             y_train = y
             X_test, y_test = None, None
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        
    return X_train, y_train, X_test, y_test, scaler

@app.route('/health', methods=['GET'])
def health_check():
    db_ok = False
    db_status_message = "disconnected"

    if db_engine:
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            db_status_message = "connected"
            db_ok = True
        except SQLAlchemyError as e:
            db_status_message = "connection_error"
            app.logger.error(f"Health check: Database connection error: {e}", exc_info=True)
    else:
        db_status_message = "not_initialized"
        app.logger.warning("Health check: Database engine not initialized.")

    if db_ok:
        return jsonify({
            "status": "success",
            "data": {
                "service_status": "healthy",
                "database_status": db_status_message
            },
            "message": "Model training service is healthy."
        }), 200
    else:
        return jsonify({
            "status": "error",
            "error": {
                "code": "SERVICE_UNHEALTHY",
                "message": "Model training service is unhealthy or database connection failed."
            },
            "details": {
                "database_status": db_status_message
            }
        }), 503

@app.route('/model/train', methods=['POST'])
def train_model_endpoint():
    if not db_engine:
        app.logger.error("Model training: Database not connected.")
        return jsonify({
            "status": "error",
            "error": {"code": "DATABASE_UNAVAILABLE", "message": "Database connection is not available."},
            "details": "Cannot fetch training data without a database connection."
        }), 503
    
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Missing JSON payload."}}), 400
    except Exception as e:
        app.logger.error(f"Invalid JSON payload for /model/train: {e}", exc_info=True)
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Invalid JSON payload."}}), 400

    required_fields = ["tickers", "start_date", "end_date", "model_name"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": f"Missing required parameters: {', '.join(missing_fields)}."}
        }), 400

    tickers = payload.get('tickers')
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    model_name = payload.get('model_name')
    
    hyperparameters = payload.get('hyperparameters', {})
    sequence_length = hyperparameters.get('sequence_length', 60)
    epochs = hyperparameters.get('num_epochs', 50)
    batch_size = hyperparameters.get('batch_size', 32)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    hidden_layer_size = hyperparameters.get('hidden_layer_size', 100)
    num_layers = hyperparameters.get('num_layers', 2) # Correctly get 'num_layers'
    test_split_size = hyperparameters.get('test_split_size', 0.2)

    # Define input_size and output_size, typically 1 for univariate time series
    input_size = 1
    output_size = 1

    if not isinstance(tickers, list) or not tickers:
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "'tickers' must be a non-empty list."}}), 400
    if not all(isinstance(t, str) for t in tickers):
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "All items in 'tickers' list must be strings."}}), 400
    # Add other specific validations for date formats, model_name format etc. if needed

    app.logger.info(f"Training request for model '{model_name}'. Tickers: {tickers}. Period: {start_date}-{end_date}")

    query_sql = text("SELECT date, ticker, adj_close FROM stock_prices WHERE ticker = ANY(:t) AND date >= :sd AND date <= :ed ORDER BY ticker, date ASC")
    
    training_summary_per_ticker = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.logger.info(f"Using device: {device}")
    
    overall_success = True # Flag to track if all tickers processed without major issues

    for ticker_symbol in tickers:
        ticker_processing_status = {"status": "initiated", "details": ""}
        try:
            app.logger.info(f"Fetching data for {ticker_symbol}...")
            with db_engine.connect() as conn:
                result = conn.execute(query_sql, {"t": [ticker_symbol], "sd": start_date, "ed": end_date}) # Use dict for params
                df_ticker = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df_ticker.empty or len(df_ticker['adj_close']) <= sequence_length:
                msg = f"Not enough data for {ticker_symbol} to create sequences (found {len(df_ticker['adj_close'])}, need > {sequence_length}). Skipping."
                app.logger.warning(msg)
                ticker_processing_status.update({"status": "skipped", "reason": "insufficient_data_for_sequences", "message": msg})
                training_summary_per_ticker[ticker_symbol] = ticker_processing_status
                continue
            
            app.logger.info(f"Preprocessing data for {ticker_symbol} ({len(df_ticker)} rows)... ")
            X_train, y_train, X_test, y_test, scaler = create_sequences_and_scale(
                df_ticker['adj_close'], sequence_length, test_split_size if test_split_size > 0 else 0
            )
            
            if len(X_train) == 0:
                msg = f"Not enough data for {ticker_symbol} after creating sequences for training. Skipping."
                app.logger.warning(msg)
                ticker_processing_status.update({"status": "skipped", "reason": "insufficient_data_post_sequencing", "message": msg})
                training_summary_per_ticker[ticker_symbol] = ticker_processing_status
                continue
                
            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            y_train_tensor = torch.from_numpy(y_train).float().to(device)
            
            X_test_tensor, y_test_tensor = None, None
            if X_test is not None and y_test is not None and len(X_test) > 0:
                X_test_tensor = torch.from_numpy(X_test).float().to(device)
                y_test_tensor = torch.from_numpy(y_test).float().to(device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # shuffle=False for time series

            model = LSTMModel(input_size=input_size, hidden_layer_size=hidden_layer_size, num_layers=num_layers, output_size=output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            app.logger.info(f"Starting training for {ticker_symbol} ({epochs} epochs)... ")
            final_train_loss = float('inf')
            for epoch in range(epochs):
                model.train()
                epoch_loss_sum = 0
                batches_processed = 0
                for seq, labels in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(seq)
                    loss = criterion(y_pred, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_sum += loss.item()
                    batches_processed +=1
                
                current_epoch_avg_loss = epoch_loss_sum / batches_processed if batches_processed > 0 else float('inf')
                final_train_loss = current_epoch_avg_loss # Update final train loss each epoch
                if (epoch + 1) % 10 == 0 or epochs <= 10 or epoch == epochs -1:
                    app.logger.info(f"Ticker: {ticker_symbol}, Epoch {epoch+1}/{epochs}, Train Loss: {current_epoch_avg_loss:.6f}")
            
            # Evaluation on test set
            test_loss_mse = None
            if X_test_tensor is not None and y_test_tensor is not None and len(X_test_tensor) > 0:
                model.eval()
                with torch.no_grad():
                    test_predictions = model(X_test_tensor)
                    test_loss_mse = criterion(test_predictions, y_test_tensor).item()
                app.logger.info(f"Ticker: {ticker_symbol}, Test Loss (MSE): {test_loss_mse:.6f}")
            else:
                app.logger.info(f"Ticker: {ticker_symbol}, No test set to evaluate.")

            # Save model and scaler
            ticker_model_dir = os.path.join(MODELS_DIR, model_name, ticker_symbol)
            os.makedirs(ticker_model_dir, exist_ok=True)
            
            model_path = os.path.join(ticker_model_dir, "model.pth")
            scaler_path = os.path.join(ticker_model_dir, "scaler.joblib")
            config_path = os.path.join(ticker_model_dir, "model_config.json")

            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
            
            # Save model configuration
            model_config = {
                'input_size': input_size,
                'hidden_layer_size': hidden_layer_size,
                'num_layers': num_layers,
                'output_size': output_size,
                'sequence_length': sequence_length
            }
            with open(config_path, 'w') as f:
                json.dump(model_config, f)

            app.logger.info(f"Model, scaler, and config saved for {ticker_symbol} at {ticker_model_dir}")
            
            ticker_processing_status.update({
                "status": "success", 
                "model_path": model_path.replace(app.root_path, '/app'),
                "scaler_path": scaler_path.replace(app.root_path, '/app'),
                "config_path": config_path.replace(app.root_path, '/app'),
                "final_train_loss_mse": final_train_loss, # Clarified MSE
                "test_loss_mse": test_loss_mse,
                "data_points_train": len(X_train_tensor),
                "data_points_test": len(X_test_tensor) if X_test_tensor is not None else 0
            })
            training_summary_per_ticker[ticker_symbol] = ticker_processing_status

        except ValueError as ve: # Specific data validation errors
            err_msg = f"ValueError for {ticker_symbol}: {ve}"
            app.logger.error(err_msg)
            ticker_processing_status.update({"status": "error", "reason": "value_error", "message": str(ve)})
            training_summary_per_ticker[ticker_symbol] = ticker_processing_status
            overall_success = False
        except Exception as e: # Catch other errors during training for a specific ticker
            err_msg = f"Error training model for {ticker_symbol}: {e}"
            app.logger.error(err_msg, exc_info=True)
            ticker_processing_status.update({"status": "error", "reason": "training_exception", "message": str(e)})
            training_summary_per_ticker[ticker_symbol] = ticker_processing_status
            overall_success = False

    # Determine overall message and status code for the endpoint
    final_http_status_code = 200
    if not overall_success and not any(t.get('status') == 'success' for t in training_summary_per_ticker.values()):
        # If all tickers failed or were skipped, maybe indicate a partial failure or different message
        final_message = f"Model training process for model '{model_name}' completed with issues for all tickers."
        # final_http_status_code = 207 # Multi-Status, or keep 200 and rely on detailed response
    elif not overall_success:
        final_message = f"Model training process for model '{model_name}' completed with some errors/skipped tickers."
    else:
        final_message = f"Model training process completed successfully for model '{model_name}'."

    return jsonify({
        "status": "success", # The endpoint call itself was successful in processing the request
        "data": {
            "model_name": model_name,
            "training_summary_per_ticker": training_summary_per_ticker
        },
        "message": final_message
    }), final_http_status_code # Typically 200, could be 207 if some sub-operations failed

@app.route('/model/predict', methods=['POST'])
def predict_model_endpoint():
    # DB check can be less stringent here if models are local
    # but might be needed if some model metadata isn't local.

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Missing JSON payload."}}), 400
    except Exception as e:
        app.logger.error(f"Invalid JSON payload for /model/predict: {e}", exc_info=True)
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Invalid JSON payload."}}), 400

    required_fields = ["model_name", "ticker", "input_sequence"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": f"Missing required parameters: {', '.join(missing_fields)}."}
        }), 400

    model_name = payload.get('model_name')
    ticker = payload.get('ticker')
    input_sequence_raw = payload.get('input_sequence')
    # hyperparameters_payload = payload.get('hyperparameters', {}) # No longer needed from payload

    if not isinstance(input_sequence_raw, list) or not all(isinstance(x, (int, float)) for x in input_sequence_raw):
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "'input_sequence' must be a list of numbers."}}), 400
    
    ticker_model_dir = os.path.join(MODELS_DIR, model_name, ticker)
    model_path = os.path.join(ticker_model_dir, "model.pth")
    scaler_path = os.path.join(ticker_model_dir, "scaler.joblib")
    config_path = os.path.join(ticker_model_dir, "model_config.json")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(config_path):
        app.logger.error(f"Model, scaler, or config not found for {model_name}/{ticker} at {ticker_model_dir}")
        return jsonify({
            "status": "error",
            "error": {"code": "MODEL_NOT_FOUND", "message": f"Model, scaler or configuration not found for {model_name}/{ticker}."}
        }), 404

    try:
        # Load model configuration
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        app.logger.info(f"Loaded model config for {model_name}/{ticker}: {model_config}")

        # Validate input sequence length against trained sequence length
        trained_sequence_length = model_config.get('sequence_length')
        if len(input_sequence_raw) != trained_sequence_length:
            msg = f"Input sequence length ({len(input_sequence_raw)}) does not match model's trained sequence length ({trained_sequence_length})."
            app.logger.error(msg)
            return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": msg}}), 400

        # Load scaler
        scaler = joblib.load(scaler_path)

        # Load model
        # Instantiate model using loaded configuration
        model = LSTMModel(
            input_size=model_config.get('input_size', 1), # Default to 1 if not in config (legacy)
            hidden_layer_size=model_config.get('hidden_layer_size', 100),
            num_layers=model_config.get('num_layers', 2),
            output_size=model_config.get('output_size', 1)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Preprocess the input sequence
        input_sequence_np = np.array(input_sequence_raw).reshape(-1, 1)
        scaled_input_sequence = scaler.transform(input_sequence_np)
        
        # Reshape for LSTM: (batch_size, sequence_length, num_features)
        # Here, batch_size is 1 as we are predicting for a single instance
        tensor_input = torch.FloatTensor(scaled_input_sequence).view(1, trained_sequence_length, model_config.get('input_size',1)).to(device)

        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(tensor_input)
        
        # Inverse transform the prediction
        predicted_adj_close = scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]

        return jsonify({
            "status": "success",
            "data": {
                "model_name": model_name,
                "ticker": ticker,
                "predicted_adj_close": float(predicted_adj_close),
                "model_config_used": model_config # Include for transparency
            },
            "message": "Prediction successful."
        }), 200
    except FileNotFoundError:
        app.logger.error(f"Model or scaler file not found during prediction for {model_name}/{ticker} at {ticker_model_dir}", exc_info=True)
        return jsonify({"status": "error", "error": {"code": "MODEL_FILE_NOT_FOUND", "message": "Model or scaler file missing."}}), 404
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding model_config.json for {model_name}/{ticker}", exc_info=True)
        return jsonify({"status": "error", "error": {"code": "CONFIG_LOAD_ERROR", "message": "Could not load/decode model configuration."}}), 500
    except Exception as e:
        app.logger.error(f"Error during prediction for {model_name}/{ticker}: {e}", exc_info=True)
        return jsonify({
            "status": "error", 
            "error": {"code": "PREDICTION_ERROR", "message": "An unexpected error occurred during prediction."},
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 