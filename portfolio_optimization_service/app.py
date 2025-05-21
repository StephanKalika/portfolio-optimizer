from flask import Flask, jsonify, request
import os
import requests # For calling other services
import pandas as pd
import numpy as np
from scipy.optimize import minimize # For portfolio optimization
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import time
import io # Added for capturing DataFrame.info() output

app = Flask(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
MODEL_TRAINING_SERVICE_URL = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://localhost:5002') # Default if not set

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

@app.route('/health', methods=['GET'])
def health_check():
    service_health_status = "healthy"
    http_status_code = 200
    details = {}

    # Check DB status
    db_ok = False
    if db_engine:
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            details["database_status"] = "connected"
            db_ok = True
        except SQLAlchemyError as e:
            details["database_status"] = "connection_error"
            app.logger.error(f"Health check: Database connection error: {e}", exc_info=True)
            service_health_status = "unhealthy"
            # http_status_code = 503 # We will set this at the end based on overall health
    else:
        details["database_status"] = "not_initialized"
        app.logger.warning("Health check: Database engine not initialized.")
        service_health_status = "unhealthy"

    # Check model_training_service status
    model_service_ok = False
    if not MODEL_TRAINING_SERVICE_URL:
        details["model_training_service_status"] = "not_configured"
        app.logger.warning("MODEL_TRAINING_SERVICE_URL is not configured.")
        service_health_status = "unhealthy"
    else:
        try:
            mts_health_url = f"{MODEL_TRAINING_SERVICE_URL}/health"
            health_resp = requests.get(mts_health_url, timeout=3) # Increased timeout slightly
            if health_resp.status_code == 200:
                mts_json = health_resp.json()
                if mts_json.get("status") == "success" and mts_json.get("data", {}).get("service_status") == "healthy":
                    details["model_training_service_status"] = "connected_and_healthy"
                    model_service_ok = True
                else:
                    details["model_training_service_status"] = "connected_but_unhealthy"
                    details["model_training_service_details"] = mts_json.get("error", {}).get("message", "No details")
                    service_health_status = "unhealthy"
            else:
                details["model_training_service_status"] = f"http_error_{health_resp.status_code}"
                service_health_status = "unhealthy"
        except requests.exceptions.Timeout:
            details["model_training_service_status"] = "unreachable_timeout"
            app.logger.warning(f"Health check: Timeout reaching model training service at {mts_health_url}")
            service_health_status = "unhealthy"
        except requests.exceptions.RequestException as e:
            details["model_training_service_status"] = "unreachable_error"
            app.logger.error(f"Health check: Error reaching model training service at {mts_health_url}: {e}", exc_info=True)
            service_health_status = "unhealthy"

    if not db_ok or not model_service_ok:
        service_health_status = "unhealthy"
        http_status_code = 503 # Service unavailable if critical dependencies are down

    if service_health_status == "healthy":
        return jsonify({
            "status": "success",
            "data": {
                "service_status": "healthy",
                **details # Unpack db_status and model_service_status
            },
            "message": "Portfolio optimization service is healthy."
        }), http_status_code
    else:
        return jsonify({
            "status": "error",
            "error": {
                "code": "SERVICE_UNHEALTHY",
                "message": "Portfolio optimization service is unhealthy or key dependencies are unavailable."
            },
            "details": details
        }), http_status_code

def get_predictions_for_assets(assets, model_name, prediction_params, historical_data_end_date):
    """Helper to get predictions for multiple assets. 
       Returns a dict where keys are asset tickers and values are dicts:
       {'status': 'success', 'prediction': value} or 
       {'status': 'error', 'code': 'ERROR_CODE', 'message': 'Details'}
    """
    app.logger.info(f"[get_predictions_for_assets] Called for assets: {assets}, model: {model_name}") # LOGGING
    prediction_results_map = {}
    sequence_length = prediction_params.get('sequence_length', 60) # Assuming default if not passed
    app.logger.info(f"[get_predictions_for_assets] Using sequence_length: {sequence_length}") # LOGGING
    
    if not db_engine:
        app.logger.error("[get_predictions_for_assets] Database not available.") # LOGGING
        # This error will be caught by the main /optimize endpoint
        raise ConnectionError("Database not available to fetch sequences for prediction.")
    if not MODEL_TRAINING_SERVICE_URL:
        app.logger.error("[get_predictions_for_assets] MODEL_TRAINING_SERVICE_URL is not configured.") # LOGGING
        # Return error status for all assets if service URL is missing
        for asset in assets:
            prediction_results_map[asset] = {
                "status": "error", 
                "code": "SERVICE_NOT_CONFIGURED", 
                "message": "Model Training Service URL not configured."
            }
        return prediction_results_map

    for asset in assets:
        app.logger.info(f"[get_predictions_for_assets] Processing asset: {asset}") # LOGGING
        try:
            query_seq = text("""
                SELECT adj_close FROM stock_prices 
                WHERE ticker = :ticker AND date <= :end_date 
                ORDER BY date DESC LIMIT :limit
            """)
            app.logger.info(f"[get_predictions_for_assets] Fetching sequence for {asset} with end_date: {historical_data_end_date}, limit: {sequence_length}") # LOGGING
            with db_engine.connect() as conn:
                result_seq = conn.execute(query_seq, {"ticker": asset, "end_date": historical_data_end_date, "limit": sequence_length})
                input_sequence_raw = [float(row[0]) for row in result_seq.fetchall()][::-1] # Reverse to get chronological order
            app.logger.info(f"[get_predictions_for_assets] Fetched sequence for {asset}, length: {len(input_sequence_raw)}") # LOGGING
            
            if len(input_sequence_raw) < sequence_length:
                msg = f"Not enough historical data ({len(input_sequence_raw)} points) for {asset} to form input sequence of length {sequence_length}."
                app.logger.warning(f"[get_predictions_for_assets] {msg}") # LOGGING
                prediction_results_map[asset] = {"status": "error", "code": "INSUFFICIENT_DATA_FOR_SEQUENCE", "message": msg}
                continue

            predict_payload = {
                "model_name": model_name,
                "ticker": asset,
                "input_sequence": input_sequence_raw,
                # hyperparameters are now read from model_config.json by model_training_service
                # "hyperparameters": prediction_params.get('hyperparameters', {}) 
            }
            
            predict_url = f"{MODEL_TRAINING_SERVICE_URL}/model/predict"
            app.logger.info(f"[get_predictions_for_assets] Calling prediction service for {asset} at {predict_url} with payload: {predict_payload}") # LOGGING
            resp = requests.post(predict_url, json=predict_payload, timeout=15) # Increased timeout
            app.logger.info(f"[get_predictions_for_assets] Prediction service response for {asset} - Status: {resp.status_code}, Body: {resp.text[:200]}...") # LOGGING (truncate body)
            
            if resp.status_code == 200:
                resp_data = resp.json()
                if resp_data.get("status") == "success":
                    prediction_results_map[asset] = {
                        "status": "success", 
                        "prediction": resp_data.get("data", {}).get('predicted_adj_close')
                    }
                else: # Model service returned success HTTP but error status in JSON
                    err_details = resp_data.get("error", {"code": "PREDICTION_SERVICE_LOGIC_ERROR", "message": "Prediction service indicated an error."}) 
                    app.logger.error(f"Prediction service returned error for {asset}: {err_details}")
                    prediction_results_map[asset] = {"status": "error", **err_details} # Unpack error details
            else: # HTTP error from prediction service
                error_code = f"PREDICTION_SERVICE_HTTP_{resp.status_code}"
                try:
                    error_message = resp.json().get("error", {}).get("message", resp.text) # Try to get JSON error message
                except ValueError: # If response is not JSON
                    error_message = resp.text[:200] # Limit length
                app.logger.error(f"HTTP error {resp.status_code} getting prediction for {asset} from {predict_url}. Message: {error_message}")
                prediction_results_map[asset] = {"status": "error", "code": error_code, "message": error_message}

        except requests.exceptions.Timeout:
            msg = f"Timeout connecting to prediction service for {asset} at {MODEL_TRAINING_SERVICE_URL}/model/predict"
            app.logger.error(msg)
            prediction_results_map[asset] = {"status": "error", "code": "PREDICTION_SERVICE_TIMEOUT", "message": msg}
        except requests.exceptions.RequestException as e: # Other connection errors
            msg = f"Error connecting to prediction service for {asset}: {e}"
            app.logger.error(msg)
            prediction_results_map[asset] = {"status": "error", "code": "PREDICTION_SERVICE_CONNECTION_ERROR", "message": msg}
        except Exception as e: # Catch-all for unexpected errors in this function for an asset
            msg = f"Unexpected error fetching prediction for {asset}: {e}"
            app.logger.error(msg, exc_info=True)
            prediction_results_map[asset] = {"status": "error", "code": "INTERNAL_PREDICTION_HELPER_ERROR", "message": msg}
            
    return prediction_results_map

def get_historical_returns_and_covariance(assets, start_date, end_date):
    """Fetches historical data, calculates daily returns, mean returns, and covariance matrix."""
    app.logger.info(f"[get_historical_returns_and_covariance] Called for assets: {assets}, period: {start_date}-{end_date}") # LOGGING
    if not db_engine:
        app.logger.error("[get_historical_returns_and_covariance] Database not available.") # LOGGING
        raise ConnectionError("Database not available for historical returns.")

    all_returns_df = pd.DataFrame()
    query = text("""
        SELECT date, ticker, adj_close FROM stock_prices
        WHERE ticker = ANY(:tickers) AND date >= :start_date AND date <= :end_date
        ORDER BY ticker, date ASC;
    """)
    with db_engine.connect() as connection:
        result = connection.execute(query, tickers=assets, start_date=start_date, end_date=end_date)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    app.logger.info(f"Inside get_historical_returns_and_covariance for assets: {assets}")
    app.logger.info(f"Raw data df.head():\n{df.head()}")
    s_io = io.StringIO()
    df.info(buf=s_io)
    app.logger.info(f"Raw data df.info():\n{s_io.getvalue()}")

    if df.empty:
        app.logger.error("No historical data found for assets in the given date range. df is empty.")
        raise ValueError("No historical data found for assets in the given date range.")

    # Pivot table to have dates as index and tickers as columns for adjusted close prices
    try:
        price_pivot = df.pivot(index='date', columns='ticker', values='adj_close')
        app.logger.info(f"Price pivot head:\n{price_pivot.head()}")
        s_io = io.StringIO()
        price_pivot.info(buf=s_io)
        app.logger.info(f"Price pivot .info():\n{s_io.getvalue()}")
        
        # Ensure all value columns are numeric before pct_change()
        for col in price_pivot.columns:
            price_pivot[col] = pd.to_numeric(price_pivot[col], errors='coerce')
        
        # Log info after conversion
        app.logger.info(f"Price pivot head after to_numeric:\n{price_pivot.head()}")
        s_io = io.StringIO()
        price_pivot.info(buf=s_io)
        app.logger.info(f"Price pivot .info() after to_numeric:\n{s_io.getvalue()}")

    except Exception as e:
        app.logger.error(f"Error during pivoting: {e}", exc_info=True)
        raise

    # Calculate daily returns
    daily_returns_unfiltered = price_pivot.pct_change()
    app.logger.info(f"Daily returns (before dropna) head:\n{daily_returns_unfiltered.head()}")
    s_io = io.StringIO()
    daily_returns_unfiltered.info(buf=s_io)
    app.logger.info(f"Daily returns (before dropna) .info():\n{s_io.getvalue()}")
    
    daily_returns = daily_returns_unfiltered.dropna()
    app.logger.info(f"Daily returns (after dropna) head:\n{daily_returns.head()}")
    app.logger.info(f"Daily returns (after dropna) shape: {daily_returns.shape}")
    app.logger.info(f"Daily returns (after dropna) empty: {daily_returns.empty}")
    s_io = io.StringIO()
    daily_returns.info(buf=s_io)
    app.logger.info(f"Daily returns (after dropna) .info():\n{s_io.getvalue()}")
    app.logger.info(f"Daily returns (after dropna) describe:\n{daily_returns.describe(include='all')}")
    
    if daily_returns.empty or len(daily_returns) < 2: # Need at least 2 periods for covariance
        app.logger.error(f"Not enough data to calculate returns or covariance after processing. daily_returns.shape: {daily_returns.shape}, daily_returns.empty: {daily_returns.empty}")
        raise ValueError("Not enough data to calculate returns or covariance.")

    mean_daily_returns = daily_returns.mean()
    app.logger.info(f"Calculated mean_daily_returns:\n{mean_daily_returns}")
    app.logger.info(f"mean_daily_returns.index: {mean_daily_returns.index}")
    
    covariance_matrix = daily_returns.cov()
    app.logger.info(f"Calculated covariance_matrix head:\n{covariance_matrix.head() if not covariance_matrix.empty else 'EMPTY DF'}")
    app.logger.info(f"Calculated covariance_matrix.index: {covariance_matrix.index}")
    app.logger.info(f"Calculated covariance_matrix.columns: {covariance_matrix.columns}")
    
    app.logger.info("[get_historical_returns_and_covariance] Successfully fetched and processed historical data.") # LOGGING
    # Log DataFrame info using string buffer
    buffer = io.StringIO()
    daily_returns.info(buf=buffer)
    df_info_str = buffer.getvalue()
    app.logger.info(f"[get_historical_returns_and_covariance] daily_returns info:\n{df_info_str}")

    return mean_daily_returns, covariance_matrix, daily_returns

# --- Portfolio Optimization Logic (Example: Maximize Sharpe Ratio) ---
def optimize_for_sharpe_ratio(num_assets, expected_returns, cov_matrix, risk_free_rate=0.02):
    app.logger.info("[optimize_for_sharpe_ratio] Called.") # LOGGING
    
    def objective_sharpe(weights):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_volatility == 0: return -np.inf # Avoid division by zero
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio # Minimize negative Sharpe ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets)) # Weights between 0 and 1
    initial_weights = np.array([1./num_assets] * num_assets)

    result = minimize(objective_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    optimized_weights = result.x
    opt_return = np.sum(expected_returns * optimized_weights)
    opt_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
    opt_sharpe = (opt_return - risk_free_rate) / opt_volatility if opt_volatility !=0 else -np.inf
    
    return optimized_weights, opt_return, opt_volatility, opt_sharpe

@app.route('/optimize', methods=['POST'])
def optimize_portfolio_endpoint():
    app.logger.info("[/optimize] Received request.") # LOGGING
    if not db_engine:
        app.logger.error("[/optimize] Database not connected.") # LOGGING
        return jsonify({
            "status": "error",
            "error": {"code": "DATABASE_UNAVAILABLE", "message": "Database connection is not available."},
            "details": "Cannot perform optimization without a database connection."
        }), 503
    
    try:
        payload = request.get_json()
        app.logger.info(f"[/optimize] Payload: {payload}") # LOGGING
        if not payload:
            app.logger.warning("[/optimize] Missing JSON payload.") # LOGGING
            return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Missing JSON payload."}}), 400
    except Exception as e:
        app.logger.error(f"[/optimize] Invalid JSON payload: {e}", exc_info=True) # LOGGING
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": f"Invalid JSON payload: {e}"}}), 400

    required_fields = ["tickers", "model_name", "start_date", "end_date"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        msg = f"Missing required parameters: {', '.join(missing_fields)}."
        app.logger.warning(f"[/optimize] Validation error: {msg}") # LOGGING
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": msg}
        }), 400

    tickers = payload.get('tickers')
    model_name = payload.get('model_name')
    # For fetching historical data for mean returns and covariance
    historical_start_date = payload.get('start_date') 
    historical_end_date = payload.get('end_date') 
    
    # Parameters for fetching sequence data for prediction (can be different, e.g. more recent)
    # For now, using the same end_date as historical_end_date for simplicity
    prediction_sequence_end_date = historical_end_date 
    
    # Default sequence length for predictions, could be part of model_config in future
    # prediction_params = payload.get('prediction_params', {'sequence_length': 60}) 
    # The 'prediction_parameters' structure was in the original user story (step 3 of portfolio_optimization_service)
    # Let's ensure we get it from the payload as intended.
    prediction_params = payload.get('prediction_parameters') # Ensure this is passed in the request
    if not prediction_params or not isinstance(prediction_params, dict) or 'sequence_length' not in prediction_params:
        msg = "'prediction_parameters' must be a dictionary and include 'sequence_length'."
        app.logger.warning(f"[/optimize] Validation error: {msg}")
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": msg}}), 400

    app.logger.info(f"[/optimize] Parsed parameters - Tickers: {tickers}, Model: {model_name}, Hist Start: {historical_start_date}, Hist End: {historical_end_date}, Pred End: {prediction_sequence_end_date}, Pred Params: {prediction_params}") # LOGGING

    if not isinstance(tickers, list) or not tickers or not all(isinstance(t, str) for t in tickers):
        msg = "'tickers' must be a non-empty list of strings."
        app.logger.warning(f"[/optimize] Validation error: {msg}") # LOGGING
        return jsonify({"status": "error", "error": {"code": "VALIDATION_ERROR", "message": msg}}), 400

    try:
        app.logger.info("[/optimize] Attempting to get predictions for assets...") # LOGGING
        predictions_map = get_predictions_for_assets(tickers, model_name, prediction_params, prediction_sequence_end_date)
        app.logger.info(f"[/optimize] Predictions map received: {predictions_map}") # LOGGING

        # Check if all predictions were successful
        successful_predictions = {asset: data['prediction'] for asset, data in predictions_map.items() if data['status'] == 'success' and data['prediction'] is not None}
        
        if len(successful_predictions) < len(tickers):
            failed_assets_details = {asset: data for asset, data in predictions_map.items() if data['status'] != 'success' or data['prediction'] is None}
            app.logger.warning(f"[/optimize] Failed to get predictions for some assets: {failed_assets_details}") # LOGGING
            # Decide if to proceed with partial data or return error
            # For now, if any prediction fails, we return an error.
            # Could be enhanced to optimize with available assets if some fail.
            return jsonify({
                "status": "error",
                "error": {"code": "PREDICTION_FAILED", "message": "Failed to get predictions for one or more assets."},
                "details": failed_assets_details
            }), 500 # Internal server error or Bad Gateway if model service is down

        # ----MODIFICATION START: Use model predictions for expected returns ----
        app.logger.info(f"[/optimize] Successfully fetched predictions: {successful_predictions}")

        # Need current prices to calculate returns from predicted prices
        current_prices_map = {}
        query_current_price = text("SELECT adj_close FROM stock_prices WHERE ticker = :ticker AND date <= :end_date ORDER BY date DESC LIMIT 1")
        
        assets_for_return_calc = list(successful_predictions.keys()) # Assets for which we have predictions
        
        with db_engine.connect() as conn:
            for asset in assets_for_return_calc:
                # Use prediction_sequence_end_date as the "current" date for price lookup
                res = conn.execute(query_current_price, {"ticker": asset, "end_date": prediction_sequence_end_date}).scalar_one_or_none()
                if res is not None:
                    current_prices_map[asset] = float(res)
                else:
                    app.logger.warning(f"[/optimize] Could not retrieve current price for {asset} (needed for predicted return calc). It will be excluded.")
        
        # Filter assets that have both prediction and current price
        assets_with_data_for_predicted_returns = [
            asset for asset in assets_for_return_calc if asset in current_prices_map and asset in successful_predictions
        ]

        if not assets_with_data_for_predicted_returns:
            app.logger.error("[/optimize] No assets remaining after attempting to fetch current prices for predicted return calculation.")
            return jsonify({
                "status": "error", 
                "error": {"code": "DATA_MISSING_FOR_PREDICTED_RETURNS", "message": "Could not fetch current prices needed to calculate returns from predictions for any asset."},
                "details": {"successful_predictions": successful_predictions, "current_prices_fetched": current_prices_map }
            }), 500

        predicted_next_day_prices_arr = np.array([successful_predictions[asset] for asset in assets_with_data_for_predicted_returns])
        current_asset_prices_arr = np.array([current_prices_map[asset] for asset in assets_with_data_for_predicted_returns])
        
        # Calculate daily returns from predictions
        expected_daily_returns_pred = (predicted_next_day_prices_arr / current_asset_prices_arr) - 1
        app.logger.info(f"[/optimize] Calculated expected daily returns from predictions: {dict(zip(assets_with_data_for_predicted_returns, expected_daily_returns_pred))}")

        # Annualize predicted daily returns
        expected_returns_for_opt = expected_daily_returns_pred * 252 # Annualized
        app.logger.info(f"[/optimize] Annualized model-predicted returns for optimization: {dict(zip(assets_with_data_for_predicted_returns, expected_returns_for_opt))}")
        
        # The assets list for historical data must now match assets_with_data_for_predicted_returns
        final_assets_for_opt = assets_with_data_for_predicted_returns
        # ----MODIFICATION END ----

        app.logger.info(f"[/optimize] Attempting to get historical returns and covariance for assets: {final_assets_for_opt}...") # LOGGING
        # Pass final_assets_for_opt to get_historical_returns_and_covariance
        # Adjust to correctly unpack the three return values from the helper function
        mean_returns_hist, cov_matrix_hist, _ = get_historical_returns_and_covariance(final_assets_for_opt, historical_start_date, historical_end_date)
        app.logger.info(f"[/optimize] Historical returns and covariance received for {final_assets_for_opt}.") # LOGGING
        
        # Covariance matrix from historical data, ensure it's aligned and annualized
        # The function get_historical_returns_and_covariance already returns daily cov_matrix.
        # We need to reindex it here if `final_assets_for_opt` is different from the original `tickers` list used before.
        # However, `get_historical_returns_and_covariance` is now called with `final_assets_for_opt`, so it should be aligned.
        annual_cov_matrix_hist = cov_matrix_hist * 252

        num_assets = len(final_assets_for_opt)
        if num_assets != len(expected_returns_for_opt) or annual_cov_matrix_hist.shape[0] != num_assets or annual_cov_matrix_hist.shape[1] != num_assets:
            err_msg = "Mismatch in asset count between final assets, predicted returns, or covariance matrix."
            app.logger.error(f"[/optimize] {err_msg} - Final Assets: {num_assets}, Pred Returns: {len(expected_returns_for_opt)}, Cov Matrix: {annual_cov_matrix_hist.shape}") # LOGGING
            return jsonify({"status": "error", "error": {"code": "DATA_ALIGNMENT_ERROR", "message": err_msg}}), 500

        # Risk-free rate from payload or default
        optimization_config = payload.get('optimization_parameters', {})
        risk_free_rate = optimization_config.get('risk_free_rate', 0.02) # Default annual risk-free rate

        app.logger.info(f"[/optimize] Attempting to optimize for Sharpe ratio with model returns for assets: {final_assets_for_opt}...") # LOGGING
        optimized_weights, opt_return, opt_volatility, opt_sharpe = optimize_for_sharpe_ratio(
            num_assets, 
            expected_returns_for_opt, # Using annualized predicted returns
            annual_cov_matrix_hist.values, # Pass cov_matrix as numpy array
            risk_free_rate
        )
        app.logger.info(f"[/optimize] Sharpe ratio optimization completed using model returns. Sharpe: {opt_sharpe}") # LOGGING

        # Prepare output
        weights_map = {asset: weight for asset, weight in zip(final_assets_for_opt, optimized_weights.tolist())}
        
        # For output, also include the original successful_predictions (raw model output)
        # and the current prices used for return calculation.
        model_output_details = {
            asset: {
                "predicted_next_price": successful_predictions.get(asset),
                "current_price_used_for_return_calc": current_prices_map.get(asset),
                "calculated_daily_return_from_prediction": ((successful_predictions.get(asset) / current_prices_map.get(asset)) - 1) if current_prices_map.get(asset) and successful_predictions.get(asset) else None
            } for asset in final_assets_for_opt
        }
        
        app.logger.info(f"[/optimize] FINAL VALUES BEFORE JSON: opt_return={opt_return}, opt_volatility={opt_volatility}, opt_sharpe={opt_sharpe}") # DEBUG LOG

        return jsonify({
            "status": "success",
            "data": {
                "optimized_assets": final_assets_for_opt,
                "optimized_weights": weights_map, 
                "portfolio_expected_annual_return_from_model": float(opt_return) if opt_return is not None else None,
                "portfolio_expected_annual_volatility_from_hist_cov": float(opt_volatility) if opt_volatility is not None else None,
                "portfolio_sharpe_ratio_from_model_er_hist_cov": float(opt_sharpe) if opt_sharpe is not None and np.isfinite(opt_sharpe) else None, # Handle potential -np.inf
                "model_inputs_for_optimization": {
                    asset: {
                        "annualized_expected_return_used": float(expected_returns_for_opt[i]) if expected_returns_for_opt[i] is not None else None,
                        "model_prediction_details": model_output_details.get(asset)
                    } for i, asset in enumerate(final_assets_for_opt)
                }
                # "covariance_matrix_used": annual_cov_matrix_hist.to_dict() # Can be large, omit for now
            },
            "message": "Portfolio optimized successfully for Sharpe ratio using model-predicted returns and historical covariance."
        }), 200

    except ConnectionError as e: # Specifically for DB issues raised by helpers
        app.logger.error(f"[/optimize] Connection error: {e}", exc_info=True) # LOGGING
        return jsonify({"status":"error", "error": {"code": "DATABASE_UNAVAILABLE", "message": str(e)}}), 503
    except ValueError as e: # For data processing errors raised by helpers
        app.logger.error(f"[/optimize] Value error: {e}", exc_info=True) # LOGGING
        return jsonify({"status":"error", "error": {"code": "DATA_PROCESSING_ERROR", "message": str(e)}}), 500
    except Exception as e:
        app.logger.error(f"[/optimize] Unexpected error during optimization: {e}", exc_info=True) # LOGGING
        return jsonify({
            "status": "error", 
            "error": {"code": "OPTIMIZATION_ERROR", "message": "An unexpected error occurred during portfolio optimization." },
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True) 