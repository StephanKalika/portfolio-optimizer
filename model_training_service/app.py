from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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
import sklearn
import joblib # For saving the scaler
import json # For saving model config
import logging
import math
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log versions of key libraries for debugging compatibility issues
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"Scikit-learn version: {sklearn.__version__}")

app = FastAPI(title="Model Training Service", version="1.0.0",
              description="Service for training time-series forecasting models.")

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models') # Save models inside the app directory for now
os.makedirs(MODELS_DIR, exist_ok=True)

db_engine = None

def initialize_db_engine(retries=5, delay=5):
    global db_engine
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set.")
        return
    for attempt in range(retries):
        try:
            current_engine = create_engine(DATABASE_URL)
            with current_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            db_engine = current_engine
            logger.info(f"Database engine connected after {attempt + 1} attempts.")
            return
        except SQLAlchemyError as e:
            logger.error(f"Error creating database engine (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1: time.sleep(delay)
            else:
                db_engine = None
                logger.critical("Failed to connect to the database after multiple retries.")

initialize_db_engine()

# --- Pydantic Models ---
class HealthResponseData(BaseModel):
    service_status: str
    database_status: str

class HealthResponse(BaseModel):
    status: str
    data: HealthResponseData
    message: str

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail

# --- Pydantic Models for /model/list ---
class ModelConfigParams(BaseModel):
    model_type: Optional[str] = None
    sequence_length: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    hidden_layer_size: Optional[int] = None
    num_layers: Optional[int] = None
    batch_size: Optional[int] = None
    test_split_size: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    nhead: Optional[int] = None # For Transformer
    creation_timestamp_utc: Optional[str] = None
    device_used_for_training: Optional[str] = None
    tickers: Optional[List[str]] = None
    class Config:
        extra = 'allow' # Allow other fields not explicitly defined

class ListedModelInfo(BaseModel):
    model_name: str
    tickers_trained_for: List[str]
    creation_date: Optional[str] = None
    config_params: Optional[ModelConfigParams] = None
    files: Optional[List[str]] = None # To list .pth and .pkl files

class ListModelsResponse(BaseModel):
    status: str = "success"
    data: List[ListedModelInfo]
    message: Optional[str] = None

# --- Pydantic Models for /model/predict ---
class PredictRequest(BaseModel):
    model_name: str = Field(..., example="market_predictor_20231026_AAPL_MSFT")
    ticker_symbol: str = Field(..., example="AAPL")
    data_points: List[float] = Field(..., min_length=1, example=[150.0, 151.0, 150.5, 155.0])

class PredictResponseData(BaseModel):
    ticker_symbol: str
    model_name: str
    predicted_value: float
    input_sequence_length_used: int
    config_loaded_from: str
    model_file_used: str
    scaler_file_used: str

class PredictResponse(BaseModel):
    status: str = "success"
    data: PredictResponseData
    message: Optional[str] = None

# --- Pydantic Models for /model/train ---
class TrainRequest(BaseModel):
    model_name: str = Field(..., example="market_predictor_20240115_AAPL_MSFT", description="A unique name for the trained model group.")
    tickers: List[str] = Field(..., min_length=1, example=["AAPL", "MSFT"], description="List of ticker symbols to train models for.")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2020-01-01", description="Start date for training data (YYYY-MM-DD).")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2023-01-01", description="End date for training data (YYYY-MM-DD).")
    model_type: str = Field(default="lstm", example="lstm", description="Type of model (lstm, gru, transformer, wavenet).")
    sequence_length: int = Field(default=60, ge=10, le=365, description="Number of past data points to use for predicting the next point.")
    epochs: int = Field(default=50, ge=1, le=1000, description="Number of training epochs.")
    learning_rate: float = Field(default=0.001, gt=0, lt=1.0, description="Learning rate for the optimizer.")
    hidden_layer_size: int = Field(default=100, ge=10, le=1024, description="Number of units in hidden layers.")
    num_layers: int = Field(default=2, ge=1, le=10, description="Number of LSTM/GRU/Transformer layers.")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for training.")
    test_split_size: float = Field(default=0.2, ge=0.0, lt=1.0, description="Fraction of data to use for testing (0 for no test set).")
    nhead: Optional[int] = Field(default=4, ge=1, description="Number of attention heads for Transformer model (if applicable).")

class TickerTrainLog(BaseModel):
    status: str # e.g., "success", "skipped", "error"
    message: Optional[str] = None
    reason: Optional[str] = None # e.g., "insufficient_data_for_sequences"
    fetch_time_seconds: Optional[float] = None
    preprocess_time_seconds: Optional[float] = None
    train_time_seconds: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_test_loss: Optional[float] = None
    model_file_path: Optional[str] = None
    scaler_file_path: Optional[str] = None
    config_file_path: Optional[str] = None # Path to ticker-specific config if ever used, or global
    error_details: Optional[str] = None
    dataset_size_total: Optional[int] = None
    training_set_size: Optional[int] = None
    test_set_size: Optional[int] = None

class TrainResponseData(BaseModel):
    model_group_name: str
    model_type_trained: str
    status_overall: str # "completed", "completed_with_errors", "failed_pre_flight"
    overall_message: str
    model_group_config_path: Optional[str] = None
    training_summary_per_ticker: Dict[str, TickerTrainLog]

class TrainResponse(BaseModel):
    status: str # "success" if request is accepted and processing starts, "error" for bad request
    data: Optional[TrainResponseData] = None
    error: Optional[ErrorDetail] = None

# --- PyTorch Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1, **kwargs):
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

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1, **kwargs):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        gru_out, _ = self.gru(input_seq, h_0)
        predictions = self.linear(gru_out[:, -1])
        return predictions
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1, nhead=4, dropout=0.1, **kwargs):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(input_size, hidden_layer_size) # Embed input_size to d_model (hidden_layer_size)
        self.pos_encoder = PositionalEncoding(hidden_layer_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_layer_size, output_size)
        self.hidden_layer_size = hidden_layer_size

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.hidden_layer_size) # Scale embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] # Get output from the last time step
        output = self.decoder(output)
        return output
        
class WaveNetModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1, kernel_size=2, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size # This is more like num_channels
        self.num_layers = num_layers # This is more like number of blocks of dilated convs
        self.output_size = output_size
        self.kernel_size = kernel_size

        self.causal_conv_input = nn.Conv1d(input_size, hidden_layer_size, kernel_size=kernel_size, padding=(kernel_size-1)*1) # Dilation 1
        
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        receptive_field = 1
        for i in range(num_layers): # num_layers here means number of blocks
            dilation = kernel_size ** i 
            self.dilated_convs.append(nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation))
            self.residual_convs.append(nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=1))
            self.skip_convs.append(nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=1)) # Or to a different skip_channels size
            receptive_field += (kernel_size - 1) * dilation
        
        self.end_conv1 = nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=1) # Can be different size
        self.end_conv2 = nn.Conv1d(hidden_layer_size, output_size, kernel_size=1)
        logger.info(f"WaveNet receptive field: {receptive_field} for kernel_size={kernel_size}, num_blocks(num_layers)={num_layers}")

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.causal_conv_input(x)
        
        skip_connections = []
        for i in range(self.num_layers):
            dilated_x = self.dilated_convs[i](x)
            
            # Gated activation unit
            # Split into two halves for filter and gate
            filter_x = torch.tanh(dilated_x)
            gate_x = torch.sigmoid(dilated_x)
            gated_x = filter_x * gate_x
            
            res_x = self.residual_convs[i](gated_x)
            x = x + res_x # Residual connection
            
            skip_x = self.skip_convs[i](gated_x)
            skip_connections.append(skip_x)

        # Sum up skip connections
        x = sum(skip_connections)
        x = torch.relu(self.end_conv1(x))
        x = self.end_conv2(x) # No activation, direct to output_size channels
        
        # Output is (batch, output_size, seq_len), take the last time step for prediction
        return x[:, :, -1] # (batch, output_size)

# Create model factory function
def create_model(model_type: str, input_size: int = 1, hidden_layer_size: int = 100, num_layers: int = 2, output_size: int = 1, **kwargs):
    """Factory function to create the specified model type with the given parameters."""
    model_type_lower = model_type.lower()
    
    if model_type_lower == 'lstm':
        return LSTMModel(input_size, hidden_layer_size, num_layers, output_size)
    elif model_type_lower == 'gru':
        return GRUModel(input_size, hidden_layer_size, num_layers, output_size)
    elif model_type_lower == 'transformer':
        nhead = kwargs.get('nhead', 4) 
        dropout = kwargs.get('dropout', 0.1)
        return TransformerModel(input_size, hidden_layer_size, num_layers, output_size, nhead=nhead, dropout=dropout)
    elif model_type_lower == 'wavenet':
        kernel_size = kwargs.get('kernel_size', 2)
        return WaveNetModel(input_size, hidden_layer_size, num_layers, output_size, kernel_size=kernel_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: lstm, gru, transformer, wavenet")

# --- Data Preprocessing Function ---
def create_sequences_and_scale(data_series: pd.Series, sequence_length: int = 60, test_size: float = 0.2):
    if not isinstance(data_series, pd.Series):
        raise TypeError("data_series must be a pandas Series.")
    if data_series.empty:
        raise ValueError("Data series is empty.")
    if len(data_series) <= sequence_length:
        raise ValueError(f"Not enough data to create sequences. Data length {len(data_series)}, sequence length {sequence_length}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Ensure data_series.values is 2D for scaler
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    if not X: # Could happen if len(scaled_data) == sequence_length
        return np.array([]), np.array([]), np.array([]), np.array([]), scaler
    
    X, y = np.array(X), np.array(y)

    if test_size > 0 and len(X) > 1 : # Ensure there's enough data to split
        # Check if test_size would result in too small test set (e.g. < 1)
        num_test_samples = int(len(X) * test_size)
        if num_test_samples == 0 and len(X) > 1: # If test_size is very small, ensure at least 1 sample if possible
            if test_size > 0: # only if user intended a test set
                 logger.warning(f"Test split size {test_size} results in 0 test samples for data of length {len(X)}. Adjusting or skipping test set if necessary.")
                 # Decide: either force 1 sample, or skip test. Forcing might be unexpected. Let's let train_test_split handle it.
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Ensure test set is not empty after split if it was intended
        if len(X_test) == 0 and num_test_samples > 0 : # If it should have had samples but doesn't
             logger.warning(f"Test set is empty after split, though {num_test_samples} were expected. This can happen with small datasets.")
             # Fallback to no test set in this weird edge case, or use all for training if test_size led to this.
             # For now, let's return what train_test_split gave. Could be problematic.
             # Consider a minimum test set size.
             if len(X) > 1 and len(X_train) < len(X): # If train got something and test got nothing, it's weird.
                pass # Let it be, train_test_split decided. User should be aware of small dataset.

    else: # No test split requested or not enough data for a meaningful split
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([]) # Ensure they are empty numpy arrays
        if test_size > 0:
            logger.info(f"Test size is {test_size} but not enough data (X length: {len(X)}) for a split. Using all data for training.")
        
    return X_train, y_train, X_test, y_test, scaler

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    db_ok = False
    db_status_message = "disconnected"

    if db_engine:
        try:
            with db_engine.connect() as connection: # This is synchronous
                connection.execute(text("SELECT 1"))
            db_status_message = "connected"
            db_ok = True
        except SQLAlchemyError as e:
            db_status_message = "connection_error"
            logger.error(f"Health check: Database connection error: {e}", exc_info=True)
    else:
        db_status_message = "not_initialized"
        logger.warning("Health check: Database engine not initialized.")

    response_data = HealthResponseData(service_status="healthy" if db_ok else "unhealthy", database_status=db_status_message)

    if db_ok:
        return HealthResponse(
            status="success",
            data=response_data,
            message="Model training service is healthy."
        )
    else:
        # Return a 503 Service Unavailable status code
        error_content = ErrorResponse(
            error=ErrorDetail(
                code="SERVICE_UNHEALTHY",
                message="Model training service is unhealthy or database connection failed."
            )
        ).dict() # FastAPI expects a dict for JSONResponse content when status_code is set
        return JSONResponse(status_code=503, content=error_content)

@app.post("/model/train", response_model=TrainResponse, tags=["Model Training"], summary="Train a new forecasting model for specified tickers")
async def train_model_endpoint(request: TrainRequest = Body(...)):
    logger.info(f"Received model training request for '{request.model_name}' with tickers: {request.tickers}")
    logger.info(f"DEBUG - Request epochs: {request.epochs}, Request dict: {request.dict()}")
    
    if not db_engine:
        logger.error("Model training: Database not connected.")
        # Use the Pydantic error model for consistent error responses
        error_detail = ErrorDetail(code="DATABASE_ERROR", message="Database connection is not available.")
        # Return a proper FastAPI response. For client errors or server errors like this, 
        # HTTPException is often preferred as it handles sending the JSON response correctly.
        raise HTTPException(status_code=503, detail=error_detail.dict())

    # Validate dates (Pydantic can also do this with custom validators or specific types like datetime.date)
    try:
        start_dt = datetime.datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(request.end_date, "%Y-%m-%d").date()
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date.")
    except ValueError as e:
        logger.error(f"Invalid date format or range: {e}. Use YYYY-MM-DD and ensure start_date < end_date.")
        error_detail = ErrorDetail(code="VALIDATION_ERROR", message=f"Invalid date format or range: {e}. Use YYYY-MM-DD.")
        raise HTTPException(status_code=400, detail=error_detail.dict())

    model_group_dir = os.path.join(MODELS_DIR, request.model_name)
    if os.path.exists(model_group_dir) and os.listdir(model_group_dir):
        logger.warning(f"Model directory '{request.model_name}' already exists and is not empty. Files might be overwritten.")
    os.makedirs(model_group_dir, exist_ok=True)
    
    training_summary: Dict[str, TickerTrainLog] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for training.")

    # Global config for the model group
    model_group_config = request.dict()
    model_group_config["creation_timestamp_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    model_group_config["device_used_for_training"] = str(device)
    model_group_config_path = os.path.join(model_group_dir, "config.json")

    overall_training_status = "completed"
    processed_tickers_count = 0
    successful_tickers_count = 0

    for ticker_symbol in request.tickers:
        ticker_log_data = {"status": "initiated", "message": "Processing started."}
        ticker_start_time = time.time()
        
        logger.info(f"Starting training process for ticker: {ticker_symbol} (model group: {request.model_name})")

        ticker_model_dir = os.path.join(model_group_dir, ticker_symbol) 
        os.makedirs(ticker_model_dir, exist_ok=True)
        
        ticker_model_path = os.path.join(ticker_model_dir, "model.pth")
        ticker_scaler_path = os.path.join(ticker_model_dir, "scaler.joblib")

        try:
            logger.info(f"Fetching data for {ticker_symbol} from {request.start_date} to {request.end_date}...")
            fetch_start_time = time.time()
            query_sql = text("SELECT date, adj_close FROM stock_prices WHERE ticker = :ticker AND date >= :start_date AND date <= :end_date ORDER BY date ASC")
            with db_engine.connect() as conn:
                df_ticker = pd.read_sql_query(query_sql, conn, params={"ticker": ticker_symbol, "start_date": request.start_date, "end_date": request.end_date})
            ticker_log_data["fetch_time_seconds"] = round(time.time() - fetch_start_time, 2)
            logger.info(f"Data fetched for {ticker_symbol}: {len(df_ticker)} rows in {ticker_log_data['fetch_time_seconds']:.2f}s.")
            ticker_log_data["dataset_size_total"] = len(df_ticker)

            if df_ticker.empty or len(df_ticker['adj_close']) <= request.sequence_length:
                msg = f"Not enough data for {ticker_symbol} (found {len(df_ticker)}, need > {request.sequence_length}). Skipping."
                logger.warning(msg)
                ticker_log_data.update({"status": "skipped", "reason": "insufficient_data_for_sequences", "message": msg})
                training_summary[ticker_symbol] = TickerTrainLog(**ticker_log_data)
                overall_training_status = "completed_with_errors"
                continue
            
            logger.info(f"Preprocessing data for {ticker_symbol}...")
            preprocess_start_time = time.time()
            X_train, y_train, X_test, y_test, scaler = create_sequences_and_scale(
                df_ticker['adj_close'], request.sequence_length, request.test_split_size
            )
            ticker_log_data["preprocess_time_seconds"] = round(time.time() - preprocess_start_time, 2)
            logger.info(f"Preprocessing for {ticker_symbol} done in {ticker_log_data['preprocess_time_seconds']:.2f}s. Train size: {len(X_train)}, Test size: {len(X_test) if X_test is not None and len(X_test)>0 else 0}")
            ticker_log_data["training_set_size"] = len(X_train)
            ticker_log_data["test_set_size"] = len(X_test) if X_test is not None else 0
            
            if len(X_train) == 0:
                msg = f"Not enough data for {ticker_symbol} after creating sequences for training. Skipping."
                logger.warning(msg)
                ticker_log_data.update({"status": "skipped", "reason": "insufficient_data_post_sequencing", "message": msg})
                training_summary[ticker_symbol] = TickerTrainLog(**ticker_log_data)
                overall_training_status = "completed_with_errors"
                continue
                
            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            y_train_tensor = torch.from_numpy(y_train).float().to(device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
            
            X_test_tensor, y_test_tensor = None, None
            if X_test is not None and y_test is not None and len(X_test) > 0:
                X_test_tensor = torch.from_numpy(X_test).float().to(device)
                y_test_tensor = torch.from_numpy(y_test).float().to(device)

            logger.info(f"Creating {request.model_type.upper()} model for {ticker_symbol}...")
            model_creation_params = {
                'input_size': 1, 
                'output_size': 1,
                'hidden_layer_size': request.hidden_layer_size,
                'num_layers': request.num_layers,
            }
            if request.model_type.lower() == 'transformer':
                 model_creation_params['nhead'] = request.nhead
            
            model = create_model(request.model_type, **model_creation_params).to(device)
            logger.info(f"{request.model_type.upper()} model for {ticker_symbol} created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate)
            
            logger.info(f"Starting training for {ticker_symbol} ({request.epochs} epochs)...")
            train_start_time = time.time()
            epoch_train_losses = []
            epoch_test_losses = [] # Store test loss per epoch

            for epoch in range(request.epochs):
                model.train()
                batch_train_losses_epoch = []
                for i, (X_batch, y_batch) in enumerate(train_loader):
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    batch_train_losses_epoch.append(loss.item())
                
                avg_epoch_train_loss = sum(batch_train_losses_epoch) / len(batch_train_losses_epoch)
                epoch_train_losses.append(avg_epoch_train_loss)
                current_test_loss_val = float('nan')

                if X_test_tensor is not None and y_test_tensor is not None:
                    model.eval()
                    with torch.no_grad():
                        y_test_pred = model(X_test_tensor)
                        current_test_loss_val = criterion(y_test_pred, y_test_tensor).item()
                        epoch_test_losses.append(current_test_loss_val)
                
                if (epoch + 1) % max(1, request.epochs // 10) == 0 or epoch == request.epochs - 1:
                    log_msg = f"Epoch {epoch+1}/{request.epochs} for {ticker_symbol} - Train Loss: {avg_epoch_train_loss:.6f}"
                    if not math.isnan(current_test_loss_val):
                        log_msg += f", Test Loss: {current_test_loss_val:.6f}"
                    logger.info(log_msg)
            
            ticker_log_data["train_time_seconds"] = round(time.time() - train_start_time, 2)
            ticker_log_data["final_train_loss"] = round(epoch_train_losses[-1], 6) if epoch_train_losses else None
            ticker_log_data["final_test_loss"] = round(epoch_test_losses[-1], 6) if epoch_test_losses and not math.isnan(epoch_test_losses[-1]) else None
            
            # Fix for "Invalid format specifier" error - properly handle None values in f-strings
            final_train_loss_str = f"{ticker_log_data['final_train_loss']:.6f}" if ticker_log_data['final_train_loss'] is not None else "N/A"
            final_test_loss_str = f"{ticker_log_data['final_test_loss']:.6f}" if ticker_log_data['final_test_loss'] is not None else "N/A"
            logger.info(f"Training for {ticker_symbol} completed in {ticker_log_data['train_time_seconds']:.2f}s. Final Train Loss: {final_train_loss_str}, Final Test Loss: {final_test_loss_str}")

            torch.save(model.state_dict(), ticker_model_path)
            joblib.dump(scaler, ticker_scaler_path)
            logger.info(f"Model saved to: {ticker_model_path}")
            logger.info(f"Scaler saved to: {ticker_scaler_path}")
            
            ticker_log_data.update({
                "status": "success", 
                "message": "Training completed and model saved.",
                "model_file_path": str(ticker_model_path),
                "scaler_file_path": str(ticker_scaler_path),
                "config_file_path": str(model_group_config_path) # Point to the main group config
            })
            successful_tickers_count += 1

        except ValueError as ve: 
            msg = f"Validation or data error during training for {ticker_symbol}: {ve}"
            logger.error(msg, exc_info=False) # Set exc_info=False for cleaner logs unless debugging
            ticker_log_data.update({"status": "error", "reason": "validation_error", "message": msg, "error_details": str(ve)})
            overall_training_status = "completed_with_errors"
        except SQLAlchemyError as dbe:
            msg = f"Database error during training for {ticker_symbol}: {dbe}"
            logger.error(msg, exc_info=True)
            ticker_log_data.update({"status": "error", "reason": "database_error", "message": msg, "error_details": str(dbe)})
            overall_training_status = "completed_with_errors"
        except Exception as e:
            msg = f"Unexpected error during training for {ticker_symbol}: {type(e).__name__} - {e}"
            logger.error(msg, exc_info=True)
            ticker_log_data.update({"status": "error", "reason": "training_exception", "message": msg, "error_details": str(e)})
            overall_training_status = "completed_with_errors"
        
        finally:
            training_summary[ticker_symbol] = TickerTrainLog(**ticker_log_data)
            processed_tickers_count +=1
            total_ticker_time = time.time() - ticker_start_time
            logger.info(f"Processing for ticker {ticker_symbol} finished in {total_ticker_time:.2f}s with status: {ticker_log_data.get('status','unknown')}")

    # Save the global model group config file
    try:
        with open(model_group_config_path, 'w') as f_cfg:
            json.dump(model_group_config, f_cfg, indent=4, sort_keys=True)
        logger.info(f"Global model group config saved to: {model_group_config_path}")
    except Exception as e:
        logger.error(f"Failed to save global model group config '{model_group_config_path}': {e}", exc_info=True)
        if overall_training_status == "completed":
            overall_training_status = "completed_with_errors" 

    final_message = f"Training process for model group '{request.model_name}' finished. Processed {processed_tickers_count}/{len(request.tickers)} tickers. Successful: {successful_tickers_count}."
    if successful_tickers_count == 0 and processed_tickers_count > 0 and len(request.tickers) > 0:
        overall_training_status = "failed"
        final_message += " No models were successfully trained."
    elif processed_tickers_count == 0 and len(request.tickers) > 0 : # Handles case where all tickers might fail very early (e.g. all skipped)
        overall_training_status = "failed"
        final_message = f"Training process for model group '{request.model_name}' attempted for {len(request.tickers)} tickers, but none were processed (e.g. all skipped due to no data)."

    logger.info(final_message)

    response_data = TrainResponseData(
        model_group_name=request.model_name,
        model_type_trained=request.model_type,
        status_overall=overall_training_status,
        overall_message=final_message,
        model_group_config_path=str(model_group_config_path) if os.path.exists(model_group_config_path) else None,
        training_summary_per_ticker=training_summary
    )
    # Based on overall_training_status, we might choose a different HTTP status code for the response
    # For now, if the request was accepted and processed (even with errors), return 200 with detailed status in body.
    return TrainResponse(status="success", data=response_data)

# --- Model Prediction Endpoint ---
@app.post("/model/predict", response_model=PredictResponse, tags=["Model Prediction"], summary="Predict next value for a ticker using a trained model")
async def predict_model_endpoint(request: PredictRequest = Body(...)):
    """
    Predicts the next value for a given ticker symbol using a specified trained model.
    Requires the model name, ticker symbol, and a sequence of recent data points.
    """
    logger.info(f"Prediction request for model '{request.model_name}', ticker '{request.ticker_symbol}'. Data points provided: {len(request.data_points)}")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_base_path = os.path.join(MODELS_DIR, request.model_name)
    # Try to find a global config first for model params
    group_config_path = os.path.join(model_base_path, "config.json")
    
    # Specific paths for ticker files
    ticker_model_file_name_v1 = f"{request.ticker_symbol}_model.pth" # e.g. AAPL_model.pth
    ticker_scaler_file_name_v1 = f"{request.ticker_symbol}_scaler.pkl" # e.g. AAPL_scaler.pkl
    
    # Path if ticker has its own sub-directory (newer structure)
    ticker_specific_dir_path = os.path.join(model_base_path, request.ticker_symbol)
    ticker_model_file_name_v2 = "model.pth"
    ticker_scaler_file_name_v2 = "scaler.pkl"
    ticker_scaler_file_name_v2_joblib = "scaler.joblib" # Add support for joblib extension
    ticker_config_file_name_v2 = "config.json"

    model_config = None
    config_path_used = None
    model_file_path = None
    scaler_file_path = None

    # Determine paths and load config
    if os.path.exists(group_config_path):
        config_path_used = group_config_path
        logger.info(f"Loading global config: {config_path_used}")
        with open(config_path_used, 'r') as f:
            model_config = json.load(f)
        
        # Check for V1 file structure (files directly under model_name dir)
        temp_model_path = os.path.join(model_base_path, ticker_model_file_name_v1)
        temp_scaler_path = os.path.join(model_base_path, ticker_scaler_file_name_v1)
        if os.path.exists(temp_model_path) and os.path.exists(temp_scaler_path):
            model_file_path = temp_model_path
            scaler_file_path = temp_scaler_path
            logger.info(f"Using V1 file structure: model={model_file_path}, scaler={scaler_file_path}")
        else: # Check for V2 file structure (files under ticker_symbol subdir)
            temp_model_path = os.path.join(ticker_specific_dir_path, ticker_model_file_name_v2)
            temp_scaler_path = os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2)
            temp_scaler_path_joblib = os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2_joblib)
            
            if os.path.exists(temp_model_path):
                model_file_path = temp_model_path
                # Check for either pkl or joblib scaler
                if os.path.exists(temp_scaler_path):
                    scaler_file_path = temp_scaler_path
                    logger.info(f"Using V2 file structure with global config: model={model_file_path}, scaler={scaler_file_path}")
                elif os.path.exists(temp_scaler_path_joblib):
                    scaler_file_path = temp_scaler_path_joblib
                    logger.info(f"Using V2 file structure with global config: model={model_file_path}, scaler={scaler_file_path} (joblib)")
                else:
                    logger.warning(f"Model file found at {temp_model_path}, but no scaler file found at either {temp_scaler_path} or {temp_scaler_path_joblib}")
            else:
                logger.warning(f"Global config found at {group_config_path}, but model/scaler files for {request.ticker_symbol} not found in expected V1 or V2 locations.")

    elif os.path.exists(ticker_specific_dir_path): # No global config, try ticker-specific config (V2 structure)
        ticker_config_path = os.path.join(ticker_specific_dir_path, "config.json")
        if os.path.exists(ticker_config_path):
            config_path_used = ticker_config_path
            logger.info(f"Loading ticker-specific config: {config_path_used}")
            with open(config_path_used, 'r') as f:
                model_config = json.load(f) # This config should have model_type, sequence_length etc.
            
            model_file_path = os.path.join(ticker_specific_dir_path, ticker_model_file_name_v2)
            # Try both scaler file extensions
            scaler_file_path_pkl = os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2)
            scaler_file_path_joblib = os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2_joblib)
            
            if os.path.exists(scaler_file_path_pkl):
                scaler_file_path = scaler_file_path_pkl
            elif os.path.exists(scaler_file_path_joblib):
                scaler_file_path = scaler_file_path_joblib
                
            logger.info(f"Using V2 file structure with ticker-specific config: model={model_file_path}, scaler={scaler_file_path}")
        else:
            logger.warning(f"Ticker specific directory {ticker_specific_dir_path} exists, but no config.json found inside.")

    if not model_config or not model_file_path or not scaler_file_path or not os.path.exists(model_file_path) or not os.path.exists(scaler_file_path):
        err_msg = f"Model '{request.model_name}' for ticker '{request.ticker_symbol}' or its components (config, model, scaler) not found."
        scaler_path_v2_joblib = os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2_joblib)
        logger.error(err_msg + f" Checked: global_cfg={group_config_path}, model_v1={os.path.join(model_base_path, ticker_model_file_name_v1)}, scaler_v1={os.path.join(model_base_path, ticker_scaler_file_name_v1)}, model_v2={os.path.join(ticker_specific_dir_path, ticker_model_file_name_v2)}, scaler_v2.pkl={os.path.join(ticker_specific_dir_path, ticker_scaler_file_name_v2)}, scaler_v2.joblib={scaler_path_v2_joblib}")
        raise HTTPException(status_code=404, detail=ErrorResponse(error=ErrorDetail(code="MODEL_NOT_FOUND", message=err_msg)).dict())

    try:
        sequence_length = model_config.get("sequence_length", 60)
        model_type = model_config.get("model_type", "lstm").lower()
        hidden_layer_size = model_config.get("hidden_layer_size", 100)
        num_layers = model_config.get("num_layers", 2)
        input_size = model_config.get("input_size", 1)
        output_size = model_config.get("output_size", 1)
        nhead = model_config.get("nhead", 4) # For Transformer
        
        logger.info(f"Loaded config for prediction: type={model_type}, seq_len={sequence_length}, hidden={hidden_layer_size}, layers={num_layers}")

        if len(request.data_points) < sequence_length:
            err_msg = f"Insufficient data points provided ({len(request.data_points)}). Model '{request.model_name}' for ticker '{request.ticker_symbol}' requires {sequence_length} points."
            logger.error(err_msg)
            raise HTTPException(status_code=400, detail=ErrorResponse(error=ErrorDetail(code="INSUFFICIENT_DATA", message=err_msg)).dict())
        
        # Use the most recent `sequence_length` points
        input_data_for_sequence = np.array(request.data_points[-sequence_length:]).reshape(-1, 1)

        scaler = joblib.load(scaler_file_path)
        logger.info(f"Scaler loaded from {scaler_file_path}")
        
        scaled_input_data = scaler.transform(input_data_for_sequence)
        
        model_creation_params_for_predict = {
            'input_size': input_size,
            'hidden_layer_size': hidden_layer_size,
            'num_layers': num_layers,
            'output_size': output_size,
        }
        if model_type == 'transformer':
            model_creation_params_for_predict['nhead'] = nhead
        # Add other relevant params from model_config if create_model expects them e.g. dropout for transformer, kernel_size for wavenet

        model_instance = create_model(model_type, **model_creation_params_for_predict)
        model_instance.load_state_dict(torch.load(model_file_path, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        logger.info(f"Model {model_type.upper()} loaded from {model_file_path} and set to eval mode on {device}.")

        # Prepare input tensor: shape should be [batch_size, sequence_length, num_features] -> [1, sequence_length, 1]
        input_tensor = torch.from_numpy(scaled_input_data).float().unsqueeze(0).to(device) 
        
        with torch.no_grad():
            predicted_scaled = model_instance(input_tensor) # Model output should be [1, output_size]
        
        # Inverse transform the prediction
        # Ensure predicted_scaled is correctly shaped for inverse_transform, typically [1, 1] for single value
        predicted_value_unscaled = scaler.inverse_transform(predicted_scaled.cpu().numpy())[0,0] # Get the scalar value
        
        logger.info(f"Prediction for {request.ticker_symbol} with model {request.model_name}: {predicted_value_unscaled:.4f}")

        return PredictResponse(
            data=PredictResponseData(
                ticker_symbol=request.ticker_symbol,
                model_name=request.model_name,
                predicted_value=float(predicted_value_unscaled),
                input_sequence_length_used=sequence_length,
                config_loaded_from=str(config_path_used),
                model_file_used=str(model_file_path),
                scaler_file_used=str(scaler_file_path)
            ),
            message="Prediction successful."
        )

    except FileNotFoundError as e:
        logger.error(f"File not found during prediction for {request.model_name} ({request.ticker_symbol}): {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=ErrorResponse(error=ErrorDetail(code="FILE_NOT_FOUND_RUNTIME", message=f"A required model component file was not found: {e.filename}")).dict())
    except joblib.externals.loky.backend.exceptions.UnpicklingError as e: # More specific error for scaler loading
        logger.error(f"Error unpickling scaler for {request.model_name} ({request.ticker_symbol}) from {scaler_file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=ErrorResponse(error=ErrorDetail(code="SCALER_LOAD_ERROR", message="Failed to load the scaler. It might be corrupted or from an incompatible version.")).dict())
    except Exception as e:
        logger.error(f"Error during prediction for {request.model_name} ({request.ticker_symbol}): {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=ErrorResponse(error=ErrorDetail(code="PREDICTION_ERROR", message=f"An unexpected error occurred during prediction: {str(e)}")).dict())

@app.get("/model/list", response_model=ListModelsResponse, tags=["Model Management"])
async def list_models_endpoint():
    logger.info("Request to list available models.")
    models_data: List[ListedModelInfo] = []
    try:
        if not os.path.exists(MODELS_DIR):
            logger.warning(f"Models directory {MODELS_DIR} does not exist.")
            return ListModelsResponse(data=[], message="Models directory not found or is not accessible.")

        # Iterate through items in MODELS_DIR, expecting each to be a model_group directory
        for model_group_name in os.listdir(MODELS_DIR):
            model_group_path = os.path.join(MODELS_DIR, model_group_name)
            if not os.path.isdir(model_group_path):
                logger.debug(f"Skipping non-directory item in MODELS_DIR: {model_group_name}")
                continue

            group_config_path = os.path.join(model_group_path, "config.json")
            group_config_data: Optional[ModelConfigParams] = None
            creation_time_iso: Optional[str] = None
            listed_tickers: List[str] = []
            model_group_files: List[str] = ["config.json"] # Start with config.json if it exists

            if os.path.exists(group_config_path):
                try:
                    with open(group_config_path, 'r') as f:
                        config_dict = json.load(f)
                    group_config_data = ModelConfigParams(**config_dict)
                    creation_time_iso = group_config_data.creation_timestamp_utc # Prefer from config
                    listed_tickers = group_config_data.tickers or [] # Prefer from config
                except json.JSONDecodeError:
                    logger.error(f"Could not decode config.json for model group {model_group_name}")
                except Exception as e: # Catch Pydantic validation errors or other issues
                    logger.error(f"Error parsing config.json for model group {model_group_name}: {e}")
            else:
                logger.warning(f"config.json not found for model group {model_group_name}. Some info will be missing.")
            
            # If creation_timestamp_utc not in config, try to get from directory ctime
            if not creation_time_iso:
                try:
                    creation_time_iso = datetime.datetime.fromtimestamp(os.path.getctime(model_group_path)).isoformat() + "Z"
                except Exception: # pylint: disable=broad-except
                    logger.warning(f"Could not determine creation time for model group {model_group_name}")
            
            # If tickers not in config, scan subdirectories (which should be ticker names)
            if not listed_tickers:
                found_tickers_from_scan = []
                for item_in_group_dir in os.listdir(model_group_path):
                    item_path = os.path.join(model_group_path, item_in_group_dir)
                    # Assume subdirectories are named after tickers and contain model.pth/scaler.joblib
                    if os.path.isdir(item_path):
                        found_tickers_from_scan.append(item_in_group_dir)
                        # Add files from this ticker's subdir
                        for f_name in os.listdir(item_path):
                            if f_name.endswith((".pth", ".joblib", ".pkl")): # .pkl for older scalers
                                model_group_files.append(f"{item_in_group_dir}/{f_name}")
                if found_tickers_from_scan:
                    listed_tickers = sorted(list(set(found_tickers_from_scan)))
            else: # Tickers were in config, list files in their respective subdirs
                 for ticker_name_from_config in listed_tickers:
                     ticker_subdir = os.path.join(model_group_path, ticker_name_from_config)
                     if os.path.isdir(ticker_subdir):
                         for f_name in os.listdir(ticker_subdir):
                            if f_name.endswith((".pth", ".joblib", ".pkl")):
                                model_group_files.append(f"{ticker_name_from_config}/{f_name}")


            models_data.append(ListedModelInfo(
                model_name=model_group_name,
                tickers_trained_for=listed_tickers,
                creation_date=creation_time_iso,
                config_params=group_config_data, # This is the Pydantic model
                files=sorted(list(set(model_group_files)))
            ))
        
        logger.info(f"Successfully listed {len(models_data)} model groups.")
        return ListModelsResponse(data=models_data, message="Model groups listed successfully.")

    except Exception as e:
        logger.error(f"Failed to list model groups: {e}", exc_info=True)
        error_detail = ErrorDetail(code="MODEL_LIST_FAILED", message=str(e))
        # Directly raise HTTPException for FastAPI to handle the response
        raise HTTPException(status_code=500, detail=error_detail.dict())


# To run the app (for local development):
# uvicorn app:app --reload --port 8000
# The Dockerfile will use a similar command without --reload.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 