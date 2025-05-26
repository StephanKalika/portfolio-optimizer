from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Union
import os
import httpx # For async HTTP calls
import pandas as pd
import numpy as np
from scipy.optimize import minimize # For portfolio optimization
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import time
import io # Added for capturing DataFrame.info() output
import logging # Use standard logging
import json # Add this import for json.JSONDecodeError in health check
import datetime # Add datetime module import to fix NameError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
# Import Prometheus monitoring
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, Summary, Info, CollectorRegistry
import psutil
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="Portfolio Optimizer Optimization Service",
    description="Service for optimizing portfolio weights based on model predictions and historical data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Portfolio Optimizer Team",
        "url": "https://github.com/StephanKalika/portfolio-optimizer",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
class PrometheusMetrics:
    def __init__(self, app_name):
        self.app_name = app_name
        self.registry = CollectorRegistry()
        
        # Request metrics - simplify labels
        self.request_counter = Counter(
            f'{app_name}_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'path', 'code'],
            registry=self.registry
        )
        
        # Request duration metrics - simplify labels
        self.request_duration = Histogram(
            f'{app_name}_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'path', 'code'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            f'{app_name}_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            f'{app_name}_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            f'{app_name}_service_info',
            'Information about the service',
            registry=self.registry
        )
        self.service_info.info({
            'version': os.environ.get('SERVICE_VERSION', 'unknown'),
            'name': app_name
        })
        
        # System metrics collection thread
        self._metrics_thread = None
        self._running = False
    
    def start_metrics_collection(self):
        """Start collecting system metrics in a background thread"""
        self._running = True
        self._metrics_thread = threading.Thread(target=self._collect_system_metrics)
        self._metrics_thread.daemon = True
        self._metrics_thread.start()
        
    def _collect_system_metrics(self):
        """Collect system metrics in a background thread"""
        while self._running:
            # Update CPU and memory metrics
            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.Process(os.getpid()).memory_info().rss)
            time.sleep(15)  # Update every 15 seconds
    
    def track_request(self, method, endpoint, status_code, duration):
        """Track an HTTP request with its duration"""
        # Ensure all values are strings and use simplified label names
        method_str = str(method)
        path_str = str(endpoint)
        code_str = str(status_code)
        
        self.request_counter.labels(method=method_str, path=path_str, code=code_str).inc()
        self.request_duration.labels(method=method_str, path=path_str, code=code_str).observe(duration)
    
    def stop(self):
        """Stop the metrics collection thread"""
        self._running = False
        if self._metrics_thread:
            self._metrics_thread.join(timeout=1.0)

class FastAPIPrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, app_name: str = "fastapi_app"):
        super().__init__(app)
        self.app = app
        self.app_name = app_name
        self.metrics = PrometheusMetrics(app_name)
        
        # Start the system metrics collection
        self.metrics.start_metrics_collection()
    
    async def dispatch(self, request: Request, call_next):
        # Handle metrics endpoint directly in the middleware
        if request.url.path == "/metrics":
            return Response(
                content=generate_latest(self.metrics.registry),
                media_type=CONTENT_TYPE_LATEST
            )
            
        # For other requests, track metrics
        start_time = time.time()
        response = await call_next(request)
        request_time = time.time() - start_time
        
        # Track request metrics with the correct label names
        self.metrics.track_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
            duration=request_time
        )
        
        return response

# Add Prometheus monitoring middleware
app.add_middleware(
    FastAPIPrometheusMiddleware,
    app_name="portfolio_optimization_service"
)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
MODEL_TRAINING_SERVICE_URL = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://model_training_service:8000')

# Initialize logger to show the current config
logger.info(f"DATABASE_URL: {DATABASE_URL}")
logger.info(f"MODEL_TRAINING_SERVICE_URL: {MODEL_TRAINING_SERVICE_URL}")

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
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                db_engine = None
                logger.critical("Failed to connect to database after multiple retries.")

initialize_db_engine()

# --- Pydantic Models ---
class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail

class HealthDependencyStatus(BaseModel):
    status: str
    details: Optional[str] = None
    url_checked: Optional[str] = None

class HealthResponseData(BaseModel):
    service_status: str
    database: HealthDependencyStatus
    model_training_service: HealthDependencyStatus

class HealthResponse(BaseModel):
    status: str # "success" or "error"
    data: HealthResponseData
    message: str

# --- Pydantic Models for /optimize ---
class OptimizeRequestParams(BaseModel):
    tickers: List[str] = Field(..., min_length=1, example=["AAPL", "MSFT"])
    model_name: str = Field(..., example="market_predictor_20240115_AAPL_MSFT")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2020-01-01", description="Start date for historical data.")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2023-01-01", description="End date for historical data & prediction sequence data.")
    prediction_parameters: Dict[str, Any] = Field(..., example={"sequence_length": 60}, description="Must include 'sequence_length'.")
    optimization_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, example={"risk_free_rate": 0.02})
    simplified_response: Optional[bool] = False

class OptimizedMetrics(BaseModel):
    portfolio_expected_annual_return_from_model: Optional[float] = None
    portfolio_expected_annual_volatility: Optional[float] = None
    portfolio_sharpe_ratio_from_model: Optional[float] = None

class PredictionBreakdownItem(BaseModel):
    predicted_next_price: Optional[float] = None
    current_price_used_for_return_calc: Optional[float] = None
    calculated_daily_return_from_prediction: Optional[float] = None

class OptimizeResponseDetails(BaseModel):
    current_prices: Dict[str, float]
    predicted_prices: Dict[str, float]
    predicted_daily_returns: Dict[str, float]
    model_predictions_breakdown: Dict[str, PredictionBreakdownItem]

class OptimizeResponseData(BaseModel):
    optimized_weights: Dict[str, float]
    metrics: OptimizedMetrics
    details: Optional[OptimizeResponseDetails] = None

class OptimizeResponse(BaseModel):
    status: str = "success"
    data: OptimizeResponseData
    message: str

# --- Helper for Async HTTP client ---
@app.on_event("startup")
async def startup_event():
    app.state.http_client = httpx.AsyncClient(timeout=10.0) # Default timeout for all calls
    logger.info("httpx.AsyncClient initialized and attached to app.state")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.http_client.aclose()
    logger.info("httpx.AsyncClient closed")

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    overall_service_status = "healthy"
    http_status_code = 200

    # Check DB status
    db_status = "disconnected"
    db_details = None
    if db_engine:
        try:
            with db_engine.connect() as connection: # Synchronous call
                connection.execute(text("SELECT 1"))
            db_status = "connected"
        except SQLAlchemyError as e:
            db_status = "connection_error"
            db_details = str(e)
            logger.error(f"Health check: Database connection error: {e}", exc_info=True)
            overall_service_status = "unhealthy"
    else:
        db_status = "not_initialized"
        logger.warning("Health check: Database engine not initialized.")
        overall_service_status = "unhealthy"
    
    db_health = HealthDependencyStatus(status=db_status, details=db_details)

    # Check model_training_service status
    mts_status = "not_configured"
    mts_details = None
    mts_url_checked = None

    if not MODEL_TRAINING_SERVICE_URL:
        logger.warning("MODEL_TRAINING_SERVICE_URL is not configured for health check.")
        overall_service_status = "unhealthy"
    else:
        mts_url_checked = str(MODEL_TRAINING_SERVICE_URL).rstrip('/') + "/health"
        try:
            client: httpx.AsyncClient = app.state.http_client
            health_resp = await client.get(mts_url_checked, timeout=5.0)
            
            if health_resp.status_code == 200:
                try:
                    mts_json = health_resp.json()
                    if mts_json.get("status") == "success" and mts_json.get("data", {}).get("service_status") == "healthy":
                        mts_status = "connected_and_healthy"
                    else:
                        mts_status = "connected_but_unhealthy"
                        mts_details = mts_json.get("error", {}).get("message", "No details from dependent service")
                        overall_service_status = "unhealthy"
                except json.JSONDecodeError as e:
                    mts_status = "invalid_response_json"
                    mts_details = f"Failed to decode JSON from MTS health: {e}. Response: {health_resp.text[:200]}"
                    logger.error(mts_details)
                    overall_service_status = "unhealthy"
        except httpx.TimeoutException:
            mts_status = "unreachable_timeout"
            logger.warning(f"Health check: Timeout reaching model training service at {mts_url_checked}")
            overall_service_status = "unhealthy"
        except httpx.RequestError as e:
            mts_status = "unreachable_error"
            mts_details = str(e)
            logger.error(f"Health check: Error reaching model training service at {mts_url_checked}: {e}", exc_info=True)
            overall_service_status = "unhealthy"

    mts_health = HealthDependencyStatus(status=mts_status, details=mts_details, url_checked=mts_url_checked)

    response_data = HealthResponseData(
        service_status=overall_service_status,
        database=db_health,
        model_training_service=mts_health
    )

    if overall_service_status == "healthy":
        return HealthResponse(
            status="success",
            data=response_data,
            message="Portfolio optimization service is healthy."
        )
    else:
        # For FastAPI, it's better to raise HTTPException for error statuses
        # rather than returning a JSONResponse directly with a 200 OK and error in body.
        # However, for /health, it's common to return 200 if the service itself is running
        # but indicate unhealthy dependencies in the body. Or 503 if critical ones are down.
        # Let's use 503 if overall_service_status is unhealthy.
        error_payload = ErrorResponse(
            error=ErrorDetail(
                code="SERVICE_UNHEALTHY", 
                message="Portfolio optimization service is unhealthy or key dependencies are unavailable."
            )
        ).dict()
        # Add the detailed health data to the error response for more context
        error_payload["details"] = response_data.dict()
        return JSONResponse(status_code=503, content=error_payload)

async def get_predictions_for_assets_async(
    http_client: httpx.AsyncClient, 
    assets: List[str], 
    model_name: str, 
    prediction_params: Dict[str, Any], 
    prediction_sequence_end_date: str
) -> Dict[str, Dict[str, Any]]:
    logger.info(f"[get_predictions_for_assets_async] Called for assets: {assets}, model: {model_name}")
    prediction_results_map = {}
    sequence_length = prediction_params.get('sequence_length')
    if not sequence_length or not isinstance(sequence_length, int) or sequence_length <=0:
        logger.error("[get_predictions_for_assets_async] Invalid or missing 'sequence_length' in prediction_params.")
        for asset in assets:
            prediction_results_map[asset] = {"status": "error", "code": "INVALID_SEQUENCE_LENGTH", "message": "Invalid sequence_length."}
        return prediction_results_map

    logger.info(f"[get_predictions_for_assets_async] Using sequence_length: {sequence_length}")
    
    if not db_engine:
        logger.error("[get_predictions_for_assets_async] Database not available.")
        raise ConnectionError("Database not available to fetch sequences for prediction.") # Will be caught by main endpoint

    if not MODEL_TRAINING_SERVICE_URL:
        logger.error("[get_predictions_for_assets_async] MODEL_TRAINING_SERVICE_URL is not configured.")
        for asset in assets:
            prediction_results_map[asset] = {"status": "error", "code": "SERVICE_NOT_CONFIGURED", "message": "Model Training Service URL not configured."}
        return prediction_results_map

    predict_url_base = str(MODEL_TRAINING_SERVICE_URL).rstrip('/')
    predict_endpoint = f"{predict_url_base}/model/predict"

    for asset in assets:
        logger.info(f"[get_predictions_for_assets_async] Processing asset: {asset}")
        try:
            query_seq = text("""
                SELECT adj_close FROM stock_prices 
                WHERE ticker = :ticker AND date <= :end_date 
                ORDER BY date DESC LIMIT :limit
            """)
            logger.info(f"[get_predictions_for_assets_async] Fetching sequence for {asset} with end_date: {prediction_sequence_end_date}, limit: {sequence_length}")
            with db_engine.connect() as conn: # Synchronous DB call
                result_seq = conn.execute(query_seq, {"ticker": asset, "end_date": prediction_sequence_end_date, "limit": sequence_length})
                # Fetchall and then reverse to get chronological order for the model
                input_sequence_raw = [float(row[0]) for row in result_seq.fetchall()][::-1] 
            logger.info(f"[get_predictions_for_assets_async] Fetched sequence for {asset}, length: {len(input_sequence_raw)}")
            
            if len(input_sequence_raw) < sequence_length:
                msg = f"Not enough historical data ({len(input_sequence_raw)} points) for {asset} to form input sequence of length {sequence_length}."
                logger.warning(f"[get_predictions_for_assets_async] {msg}")
                prediction_results_map[asset] = {"status": "error", "code": "INSUFFICIENT_DATA_FOR_SEQUENCE", "message": msg}
                continue

            predict_payload = {
                "model_name": model_name,
                "ticker_symbol": asset, # Changed from "ticker" to "ticker_symbol" to match model_training_service
                "data_points": input_sequence_raw 
            }
            
            logger.info(f"[get_predictions_for_assets_async] Calling prediction service for {asset} at {predict_endpoint} with payload body: {predict_payload}")
            resp = await http_client.post(predict_endpoint, json=predict_payload, timeout=20.0) # Increased timeout for model prediction
            logger.info(f"[get_predictions_for_assets_async] Prediction service response for {asset} - Status: {resp.status_code}, Body: {resp.text[:200]}...")
            
            resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            resp_data = resp.json() # Assuming model_training_service returns JSON

            if resp_data.get("status") == "success" and resp_data.get("data", {}).get("predicted_value") is not None:
                    prediction_results_map[asset] = {
                        "status": "success", 
                    "prediction": resp_data["data"]["predicted_value"],
                    "details": resp_data.get("data") # Store full data part of prediction response
                }
            else:
                err_code = resp_data.get("error", {}).get("code", "PREDICTION_SERVICE_LOGIC_ERROR")
                err_msg = resp_data.get("error", {}).get("message", "Prediction service indicated an error or missing data.")
                logger.error(f"Prediction service returned non-success or no value for {asset}: Code={err_code}, Msg={err_msg}")
                prediction_results_map[asset] = {"status": "error", "code": err_code, "message": err_msg}

        except httpx.TimeoutException:
            msg = f"Timeout connecting to prediction service for {asset} at {predict_endpoint}"
            logger.error(msg)
            prediction_results_map[asset] = {"status": "error", "code": "PREDICTION_SERVICE_TIMEOUT", "message": msg}
        except httpx.RequestError as e:
            msg = f"Error connecting to prediction service for {asset}: {e}"
            logger.error(msg, exc_info=True)
            prediction_results_map[asset] = {"status": "error", "code": "PREDICTION_SERVICE_CONNECTION_ERROR", "message": msg}
        except json.JSONDecodeError as e:
            msg = f"Failed to decode JSON response from prediction service for {asset}: {e}. Response: {resp.text[:200]}"
            logger.error(msg)
            prediction_results_map[asset] = {"status": "error", "code": "PREDICTION_SERVICE_INVALID_JSON", "message": msg}
        except Exception as e:
            msg = f"Unexpected error fetching prediction for {asset}: {type(e).__name__} - {e}"
            logger.error(msg, exc_info=True)
            prediction_results_map[asset] = {"status": "error", "code": "INTERNAL_PREDICTION_HELPER_ERROR", "message": msg}
            
    return prediction_results_map

def get_historical_returns_and_covariance(assets, start_date, end_date):
    """Fetches historical data, calculates daily returns, mean returns, and covariance matrix."""
    logger.info(f"[get_historical_returns_and_covariance] Called for assets: {assets}, period: {start_date}-{end_date}") # LOGGING
    if not db_engine:
        logger.error("[get_historical_returns_and_covariance] Database not available.") # LOGGING
        raise ConnectionError("Database not available for historical returns.")

    all_returns_df = pd.DataFrame()
    query = text("""
        SELECT date, ticker, adj_close FROM stock_prices
        WHERE ticker = ANY(:tickers) AND date >= :start_date AND date <= :end_date
        ORDER BY ticker, date ASC;
    """)
    with db_engine.connect() as connection:
        result = connection.execute(query, {"tickers": assets, "start_date": start_date, "end_date": end_date})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    logger.info(f"Inside get_historical_returns_and_covariance for assets: {assets}")
    logger.info(f"Raw data df.head():\n{df.head()}")
    s_io = io.StringIO()
    df.info(buf=s_io)
    logger.info(f"Raw data df.info():\n{s_io.getvalue()}")

    if df.empty:
        logger.error("No historical data found for assets in the given date range. df is empty.")
        raise ValueError("No historical data found for assets in the given date range.")

    # Pivot table to have dates as index and tickers as columns for adjusted close prices
    try:
        price_pivot = df.pivot(index='date', columns='ticker', values='adj_close')
        logger.info(f"Price pivot head:\n{price_pivot.head()}")
        s_io = io.StringIO()
        price_pivot.info(buf=s_io)
        logger.info(f"Price pivot .info():\n{s_io.getvalue()}")
        
        # Ensure all value columns are numeric before pct_change()
        for col in price_pivot.columns:
            price_pivot[col] = pd.to_numeric(price_pivot[col], errors='coerce')
        
        # Log info after conversion
        logger.info(f"Price pivot head after to_numeric:\n{price_pivot.head()}")
        s_io = io.StringIO()
        price_pivot.info(buf=s_io)
        logger.info(f"Price pivot .info() after to_numeric:\n{s_io.getvalue()}")

    except Exception as e:
        logger.error(f"Error during pivoting: {e}", exc_info=True)
        raise

    # Calculate daily returns
    daily_returns_unfiltered = price_pivot.pct_change()
    logger.info(f"Daily returns (before dropna) head:\n{daily_returns_unfiltered.head()}")
    s_io = io.StringIO()
    daily_returns_unfiltered.info(buf=s_io)
    logger.info(f"Daily returns (before dropna) .info():\n{s_io.getvalue()}")
    
    daily_returns = daily_returns_unfiltered.dropna()
    logger.info(f"Daily returns (after dropna) head:\n{daily_returns.head()}")
    logger.info(f"Daily returns (after dropna) shape: {daily_returns.shape}")
    logger.info(f"Daily returns (after dropna) empty: {daily_returns.empty}")
    s_io = io.StringIO()
    daily_returns.info(buf=s_io)
    logger.info(f"Daily returns (after dropna) .info():\n{s_io.getvalue()}")
    logger.info(f"Daily returns (after dropna) describe:\n{daily_returns.describe(include='all')}")
    
    if daily_returns.empty or len(daily_returns) < 2: # Need at least 2 periods for covariance
        logger.error(f"Not enough data to calculate returns or covariance after processing. daily_returns.shape: {daily_returns.shape}, daily_returns.empty: {daily_returns.empty}")
        raise ValueError("Not enough data to calculate returns or covariance.")

    mean_daily_returns = daily_returns.mean()
    logger.info(f"Calculated mean_daily_returns:\n{mean_daily_returns}")
    logger.info(f"mean_daily_returns.index: {mean_daily_returns.index}")
    
    covariance_matrix = daily_returns.cov()
    logger.info(f"Calculated covariance_matrix head:\n{covariance_matrix.head() if not covariance_matrix.empty else 'EMPTY DF'}")
    logger.info(f"Calculated covariance_matrix.index: {covariance_matrix.index}")
    logger.info(f"Calculated covariance_matrix.columns: {covariance_matrix.columns}")
    
    logger.info("[get_historical_returns_and_covariance] Successfully fetched and processed historical data.") # LOGGING
    # Log DataFrame info using string buffer
    buffer = io.StringIO()
    daily_returns.info(buf=buffer)
    df_info_str = buffer.getvalue()
    logger.info(f"[get_historical_returns_and_covariance] daily_returns info:\n{df_info_str}")

    return mean_daily_returns, covariance_matrix, daily_returns

# --- Portfolio Optimization Logic (Example: Maximize Sharpe Ratio) ---
def optimize_for_sharpe_ratio(num_assets, expected_returns, cov_matrix, risk_free_rate=0.02):
    logger.info("[optimize_for_sharpe_ratio] Called.") # LOGGING
    
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

@app.post("/optimize", response_model=OptimizeResponse, tags=["Portfolio Optimization"], summary="Optimize portfolio based on model predictions and historical data")
async def optimize_portfolio_endpoint(request: OptimizeRequestParams = Body(...)):
    logger.info(f"[/optimize] Received request: {request.model_dump_json(indent=2)}")
    
    if not db_engine:
        logger.error("[/optimize] Database not connected.")
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DATABASE_UNAVAILABLE", message="Database connection not available.").dict())
    
    # Validate dates (already patterned in Pydantic, but good to double check range)
    try:
        start_dt = datetime.datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(request.end_date, "%Y-%m-%d").date()
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date.")
    except ValueError as e:
        logger.error(f"[/optimize] Invalid date range: {e}")
        raise HTTPException(status_code=400, detail=ErrorDetail(code="VALIDATION_ERROR", message=str(e)).dict())

    if not request.prediction_parameters.get("sequence_length") or not isinstance(request.prediction_parameters["sequence_length"], int):
        logger.error("[/optimize] 'sequence_length' missing or invalid in prediction_parameters.")
        raise HTTPException(status_code=400, detail=ErrorDetail(code="VALIDATION_ERROR", message="'sequence_length' must be an integer in prediction_parameters.").dict())

    http_client: httpx.AsyncClient = app.state.http_client

    try:
        logger.info("[/optimize] Attempting to get predictions for assets...")
        predictions_map = await get_predictions_for_assets_async(
            http_client,
            request.tickers,
            request.model_name,
            request.prediction_parameters,
            request.end_date # prediction_sequence_end_date is same as hist_end_date
        )
        logger.info(f"[/optimize] Predictions map received: {predictions_map}")

        successful_predictions_data = {asset: data for asset, data in predictions_map.items() if data.get('status') == 'success' and data.get('prediction') is not None}
        
        if len(successful_predictions_data) < len(request.tickers):
            failed_assets_details = {asset: data for asset, data in predictions_map.items() if data.get('status') != 'success' or data.get('prediction') is None}
            logger.warning(f"[/optimize] Failed to get predictions for some assets: {failed_assets_details}")
            # Consider if partial optimization is allowed or if it should be all-or-nothing
            raise HTTPException(status_code=500, # Or 422 if it's more like a processing error due to bad upstream data
                detail=ErrorDetail(code="PREDICTION_FAILED_PARTIAL", 
                                   message="Failed to get valid predictions for one or more assets.").dict(),
                headers={"X-Failed-Assets-Details": json.dumps(failed_assets_details)} # Custom header for more info
            )
        
        successful_predictions_values = {asset: data['prediction'] for asset, data in successful_predictions_data.items()}
        logger.info(f"[/optimize] Successfully fetched predictions values: {successful_predictions_values}")

        current_prices_map = {}
        query_current_price = text("SELECT adj_close FROM stock_prices WHERE ticker = :ticker AND date <= :end_date ORDER BY date DESC LIMIT 1")
        assets_for_return_calc = list(successful_predictions_values.keys())
        
        with db_engine.connect() as conn: # Synchronous DB call
            for asset in assets_for_return_calc:
                res = conn.execute(query_current_price, {"ticker": asset, "end_date": request.end_date}).scalar_one_or_none()
                if res is not None:
                    current_prices_map[asset] = float(res)
                else:
                    logger.warning(f"[/optimize] Could not retrieve current price for {asset}. It will be excluded from predicted return calculation.")
        
        assets_with_data_for_predicted_returns = [
            asset for asset in assets_for_return_calc if asset in current_prices_map
        ]

        if not assets_with_data_for_predicted_returns:
            logger.error("[/optimize] No assets remaining after attempting to fetch current prices for predicted return calculation.")
            raise HTTPException(status_code=500,
                detail=ErrorDetail(code="DATA_MISSING_FOR_PREDICTED_RETURNS", 
                                   message="Could not fetch current prices for any asset with a successful prediction.").dict(),
                headers={"X-Successful-Predictions": json.dumps(successful_predictions_values), "X-Current-Prices-Attempted": json.dumps(current_prices_map)}
            )

        predicted_next_day_prices_arr = np.array([successful_predictions_values[asset] for asset in assets_with_data_for_predicted_returns])
        current_asset_prices_arr = np.array([current_prices_map[asset] for asset in assets_with_data_for_predicted_returns])
        
        expected_daily_returns_pred = (predicted_next_day_prices_arr / current_asset_prices_arr) - 1
        logger.info(f"[/optimize] Calculated expected daily returns from predictions for {len(assets_with_data_for_predicted_returns)} assets.")

        expected_returns_for_opt = expected_daily_returns_pred * 252 # Annualized
        final_assets_for_opt = assets_with_data_for_predicted_returns # Assets used from this point on

        logger.info(f"[/optimize] Attempting to get historical returns and covariance for assets: {final_assets_for_opt}...")
        # This function is still synchronous
        mean_returns_hist, cov_matrix_hist, daily_returns_df_hist = get_historical_returns_and_covariance(
            final_assets_for_opt, request.start_date, request.end_date
        )
        logger.info(f"[/optimize] Historical returns and covariance received for {len(final_assets_for_opt)} assets.")
        
        annual_cov_matrix_hist = cov_matrix_hist * 252
        num_assets = len(final_assets_for_opt)

        if num_assets == 0: # Should have been caught earlier, but as a safeguard
            logger.error("[/optimize] No assets available for optimization after all filtering steps.")
            raise HTTPException(status_code=500, detail=ErrorDetail(code="NO_ASSETS_FOR_OPTIMIZATION", message="No assets remained for optimization.").dict())

        if num_assets != len(expected_returns_for_opt) or annual_cov_matrix_hist.shape != (num_assets, num_assets):
            err_msg = f"Mismatch in asset count or matrix dimensions. Assets: {num_assets}, Pred Returns: {len(expected_returns_for_opt)}, Cov Matrix: {annual_cov_matrix_hist.shape}"
            logger.error(f"[/optimize] {err_msg}")
            raise HTTPException(status_code=500, detail=ErrorDetail(code="DATA_ALIGNMENT_ERROR", message=err_msg).dict())

        risk_free_rate = request.optimization_parameters.get('risk_free_rate', 0.02)

        logger.info(f"[/optimize] Attempting to optimize for Sharpe ratio with model returns for assets: {final_assets_for_opt}...")
        # This function is synchronous
        optimized_weights_arr, opt_return, opt_volatility, opt_sharpe = optimize_for_sharpe_ratio(
            num_assets, 
            expected_returns_for_opt, 
            annual_cov_matrix_hist.values, # Ensure it's a NumPy array for scipy
            risk_free_rate
        )
        logger.info(f"[/optimize] Sharpe ratio optimization completed. Sharpe: {opt_sharpe:.4f}")

        weights_map = {asset: float(weight) for asset, weight in zip(final_assets_for_opt, optimized_weights_arr.tolist())}
        
        metrics = OptimizedMetrics(
            portfolio_expected_annual_return_from_model=float(opt_return) if not np.isnan(opt_return) else None,
            portfolio_expected_annual_volatility=float(opt_volatility) if not np.isnan(opt_volatility) else None,
            portfolio_sharpe_ratio_from_model=float(opt_sharpe) if not np.isnan(opt_sharpe) and np.isfinite(opt_sharpe) else None
        )

        details_data = None
        if not request.simplified_response:
            model_output_details_dict = {}
            for asset in final_assets_for_opt:
                pred_val = successful_predictions_values.get(asset)
                curr_price = current_prices_map.get(asset)
                daily_ret_pred = None
                if pred_val is not None and curr_price is not None and curr_price != 0:
                    daily_ret_pred = (pred_val / curr_price) - 1
                
                model_output_details_dict[asset] = PredictionBreakdownItem(
                    predicted_next_price=float(pred_val) if pred_val is not None else None,
                    current_price_used_for_return_calc=float(curr_price) if curr_price is not None else None,
                    calculated_daily_return_from_prediction=float(daily_ret_pred) if daily_ret_pred is not None else None
                )
            
            details_data = OptimizeResponseDetails(
                current_prices={asset: float(current_prices_map[asset]) for asset in final_assets_for_opt if asset in current_prices_map},
                predicted_prices={asset: float(successful_predictions_values[asset]) for asset in final_assets_for_opt if asset in successful_predictions_values},
                predicted_daily_returns={asset: float(ret) for asset, ret in zip(final_assets_for_opt, expected_daily_returns_pred.tolist())},
                model_predictions_breakdown=model_output_details_dict
            )

        response_data_obj = OptimizeResponseData(
            optimized_weights=weights_map, 
            metrics=metrics,
            details=details_data
        )
        
        return OptimizeResponse(data=response_data_obj, message="Portfolio optimized successfully.")

    except ConnectionError as e:
        logger.error(f"[/optimize] Database connection error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DATABASE_UNAVAILABLE_OPTIMIZE", message=str(e)).dict())
    except ValueError as e: # Includes errors from optimize_for_sharpe_ratio if not result.success
        logger.error(f"[/optimize] Value error during optimization: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=ErrorDetail(code="OPTIMIZATION_VALUE_ERROR", message=str(e)).dict())
    except Exception as e:
        logger.error(f"[/optimize] Unexpected error during optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=ErrorDetail(code="OPTIMIZATION_UNEXPECTED_ERROR", message=f"An unexpected error occurred: {str(e)}").dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True) 