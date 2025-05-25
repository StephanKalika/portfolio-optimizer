from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response as FastAPIResponse, StreamingResponse
import httpx
import os
import pybreaker
import msgpack # Keep if msgpack handling is desired
import time
import logging
import json # For JSONDecodeError and json.dumps
from functools import wraps
from typing import Dict, Any, Optional, List, Tuple, Union
from pydantic import BaseModel, HttpUrl
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import datetime
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, Summary, Info, CollectorRegistry
import psutil
import threading

# Configure logging (FastAPI uses uvicorn logging by default, but good to have a logger instance)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="Portfolio Optimizer API Gateway",
    description="API Gateway for the Portfolio Optimization System that routes requests to appropriate microservices",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    contact={
        "name": "Portfolio Optimizer Team",
        "url": "https://github.com/StephanKalika/portfolio-optimizer",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware to allow cross-origin requests
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
        
        # Request metrics
        self.request_counter = Counter(
            f'{app_name}_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'path', 'code'],
            registry=self.registry
        )
        
        # Request duration metrics
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
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f'{app_name}_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service'],
            registry=self.registry
        )
        
        # Proxy metrics
        self.proxy_requests = Counter(
            f'{app_name}_proxy_requests_total',
            'Total number of proxied requests',
            ['service', 'status'],
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
            
            # Update circuit breaker states
            for service_name, breaker in SERVICE_BREAKERS.items():
                state_value = 0  # closed (default)
                if hasattr(breaker.current_state, 'name'):
                    state_name = breaker.current_state.name
                    if state_name == "open":
                        state_value = 1
                    elif state_name == "half-open":
                        state_value = 2
                else:
                    # Handle case where current_state is a string
                    if breaker.current_state == "open":
                        state_value = 1
                    elif breaker.current_state == "half-open":
                        state_value = 2
                
                self.circuit_breaker_state.labels(service=service_name).set(state_value)
                
            time.sleep(15)  # Update every 15 seconds
    
    def track_request(self, method, endpoint, status_code, duration):
        """Track an HTTP request with its duration"""
        # Ensure all values are strings
        method_str = str(method)
        path_str = str(endpoint)
        code_str = str(status_code)
        
        self.request_counter.labels(method=method_str, path=path_str, code=code_str).inc()
        self.request_duration.labels(method=method_str, path=path_str, code=code_str).observe(duration)
    
    def track_proxy_request(self, service, status):
        """Track a proxied request"""
        self.proxy_requests.labels(service=service, status=status).inc()
    
    def stop(self):
        """Stop the metrics collection thread"""
        self._running = False
        if self._metrics_thread:
            self._metrics_thread.join(timeout=1.0)

# Initialize Prometheus metrics
prometheus_metrics = PrometheusMetrics("api_gateway_service")
prometheus_metrics.start_metrics_collection()

# Add middleware to track request durations
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Calculate request duration
    duration = time.time() - start_time
    
    # Get the route path or the raw path if no route matched
    path = request.url.path
    
    # Track the request in Prometheus
    prometheus_metrics.track_request(
        method=request.method,
        endpoint=path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return FastAPIResponse(
        content=generate_latest(prometheus_metrics.registry),
        media_type=CONTENT_TYPE_LATEST
    )

# Environment variables to locate other services
DATA_INGESTION_SERVICE_URL_STR = os.getenv('DATA_INGESTION_SERVICE_URL', 'http://data_ingestion_service:5001')
MODEL_TRAINING_SERVICE_URL_STR = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://model_training_service:8000') # Updated port
PORTFOLIO_OPTIMIZATION_SERVICE_URL_STR = os.getenv('PORTFOLIO_OPTIMIZATION_SERVICE_URL', 'http://portfolio_optimization_service:8001') # Updated port

# Parse URLs with Pydantic for validation
SERVICE_URLS: Dict[str, Optional[HttpUrl]] = {}
for service_name, url_str in [
    ("data_ingestion", DATA_INGESTION_SERVICE_URL_STR),
    ("model_training", MODEL_TRAINING_SERVICE_URL_STR),
    ("portfolio_optimization", PORTFOLIO_OPTIMIZATION_SERVICE_URL_STR)
]:
    if url_str:
        try:
            SERVICE_URLS[service_name] = HttpUrl(url_str)
        except Exception as e:
            logger.error(f"Invalid URL for {service_name}: {url_str}. Error: {e}")
            SERVICE_URLS[service_name] = None
    else:
        logger.warning(f"{service_name.upper()}_SERVICE_URL not set.")
        SERVICE_URLS[service_name] = None

# Circuit breaker configuration (remains the same)
circuit_breaker_config = {
    "failure_threshold": 5,
    "recovery_timeout": 30,
}

# Custom circuit breaker listeners for logging (no change needed for async context)
class CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    def __init__(self, service_name):
        self.service_name = service_name

    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Circuit breaker for {self.service_name} state changed from {old_state.name} to {new_state.name}")

    def failure(self, cb, exc):
        logger.error(f"Circuit breaker for {self.service_name} recorded a failure: {type(exc).__name__} - {exc}")

    def success(self, cb):
        logger.info(f"Circuit breaker for {self.service_name} recorded a success")

SERVICE_BREAKERS: Dict[str, pybreaker.CircuitBreaker] = {}
for service_name_key in ["data_ingestion", "model_training", "portfolio_optimization"]:
    SERVICE_BREAKERS[service_name_key] = pybreaker.CircuitBreaker(
    fail_max=circuit_breaker_config["failure_threshold"],
    reset_timeout=circuit_breaker_config["recovery_timeout"],
        exclude=[httpx.ConnectError, httpx.TimeoutException],
        listeners=[CircuitBreakerListener(f"{service_name_key.replace('_', ' ').title()} Service")]
    )

# --- Pydantic Models ---
class ErrorDetail(BaseModel):
    code: str
    message: str
    target_service_url: Optional[str] = None
    circuit_state: Optional[str] = None
    original_exception: Optional[str] = None
    upstream_status_code: Optional[int] = None

class GatewayErrorResponse(BaseModel):
    status: str = "error"
    source: str # "api_gateway" or "upstream"
    error: ErrorDetail

class CircuitBreakerStates(BaseModel):
    data_ingestion: str
    model_training: str
    portfolio_optimization: str

class HealthResponse(BaseModel):
    status: str = "success"
    message: str
    circuit_breaker_status: CircuitBreakerStates

# --- Helper for Async HTTP client ---
@app.on_event("startup")
async def startup_event():
    app.state.http_client = httpx.AsyncClient(timeout=20.0) 
    logger.info("httpx.AsyncClient initialized and attached to app.state")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.http_client.aclose()
    logger.info("httpx.AsyncClient closed")

# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Gateway Health"])
async def health_check_endpoint():
    def get_breaker_state_name(breaker_key: str) -> str:
        breaker = SERVICE_BREAKERS.get(breaker_key)
        if breaker:
            state = breaker.current_state
            if hasattr(state, 'name'):
                return state.name
            # If state is already a string (which is unexpected but matches the error)
            elif isinstance(state, str): 
                return state 
            return "unknown_state_object" # Should not happen
        return "breaker_not_found"

    circuit_status = CircuitBreakerStates(
        data_ingestion=get_breaker_state_name("data_ingestion"),
        model_training=get_breaker_state_name("model_training"),
        portfolio_optimization=get_breaker_state_name("portfolio_optimization")
    )
    return HealthResponse(
        message="API Gateway is healthy and running.",
        circuit_breaker_status=circuit_status
    )

# --- Proxying Logic ---
async def optimize_response_async(upstream_response: httpx.Response, original_request: Request) -> FastAPIResponse:
    content_type = upstream_response.headers.get("content-type", "").lower()
    # Preserve most headers, but exclude hop-by-hop headers that shouldn't be proxied.
    excluded_headers = ["content-encoding", "transfer-encoding", "connection", "keep-alive", 
                        "proxy-authenticate", "proxy-authorization", "te", "trailers", 
                        "upgrade", "server", "date"]
    response_headers = {name: value for name, value in upstream_response.headers.items() 
                        if name.lower() not in excluded_headers}

    # msgpack handling (if client accepts and upstream sends it)
    # Ensure msgpack is installed: pip install msgpack, and it's in requirements.txt
    if "application/x-msgpack" in content_type and "application/x-msgpack" in original_request.headers.get("accept", "").lower():
        logger.debug(f"Proxying msgpack response from upstream for: {original_request.url.path}")
        # For msgpack, we pass content directly, so upstream Content-Length should be fine IF NOT re-encoding.
        # However, if there's any doubt or transformation, recalculating C-L would be safer.
        # For now, assuming direct pass-through, keep original C-L if present.
        return FastAPIResponse(content=upstream_response.content, 
                               status_code=upstream_response.status_code, 
                               headers=response_headers, 
                               media_type="application/x-msgpack")

    if "application/json" in content_type:
        # If we are re-parsing and re-serializing JSON, we must let the new JSONResponse
        # calculate its own Content-Length based on its rendering.
        # Remove any Content-Length from the headers copied from upstream.
        response_headers_for_json = {k: v for k, v in response_headers.items() if k.lower() != 'content-length'}
        try:
            # For FastAPIResponse with JSON, pass the decoded content directly.
            # httpx.Response.json() handles decoding.
            json_data = upstream_response.json()
            logger.debug(f"Proxying JSON response from upstream for: {original_request.url.path}")
            return JSONResponse(content=json_data, 
                                status_code=upstream_response.status_code, 
                                headers=response_headers_for_json) # Use headers without upstream C-L
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from upstream for {original_request.url.path}: {e}. Response text (truncated): {upstream_response.text[:200]}")
            # If JSON decoding fails for a response that claims to be JSON, return as plain text to avoid further errors.
            # Ensure Content-Length is removed here too, as we are changing content representation.
            response_headers_for_plain_text = {k: v for k, v in response_headers.items() if k.lower() != 'content-length'}
            response_headers_for_plain_text["content-type"] = "text/plain" # Correct content type
            return FastAPIResponse(content=upstream_response.content, # Sending original bytes
                                   status_code=upstream_response.status_code, 
                                   headers=response_headers_for_plain_text, 
                                   media_type="text/plain") # Starlette will set C-L for these bytes
    else:
        # For other content types (e.g., text/html, images, etc.), stream the content.
        # StreamingResponse is suitable for this.
        # If Content-Length is in response_headers, StreamingResponse will use it.
        # This is generally okay if we are truly streaming bytes as-is.
        logger.debug(f"Proxying non-JSON/msgpack response (type: {content_type}) as stream for: {original_request.url.path}")
        return StreamingResponse(upstream_response.aiter_bytes(), 
                                 status_code=upstream_response.status_code, 
                                 headers=response_headers, 
                                 media_type=upstream_response.headers.get("content-type")) # Use original content-type

async def proxy_request_async(
    service_name_key: str, 
    original_request: Request, 
    path_suffix: str, 
    timeout: Optional[float] = None, # Allow overriding default client timeout
) -> FastAPIResponse:
    
    # Get the target service URL 
    service_url = SERVICE_URLS.get(service_name_key)
    if not service_url:
        logger.error(f"Service URL for '{service_name_key}' is not configured.")
        error_detail = ErrorDetail(
            code="SERVICE_NOT_CONFIGURED", 
            message=f"Service '{service_name_key}' is not properly configured in the API Gateway.",
            target_service_url=None
        )
        # Track proxy request failure
        prometheus_metrics.track_proxy_request(service_name_key, "error")
        return JSONResponse(
            status_code=500,
            content=GatewayErrorResponse(
                source="api_gateway",
                error=error_detail
            ).dict()
        )
    
    # Ensure proper URL concatenation - strip any leading slash from path_suffix and ensure base URL has trailing slash
    base_url = str(service_url).rstrip('/') + '/'
    path = path_suffix.lstrip('/')
    target_url = f"{base_url}{path}"
    
    logger.info(f"Proxying {original_request.method} to {target_url} for service '{service_name_key}' with timeout {timeout}")
    
    try:
        # Use the circuit breaker for the service
        breaker = SERVICE_BREAKERS[service_name_key]

        @breaker
        async def make_request_inner_async(): 
            request_body = await original_request.body() # Get the raw body bytes

            headers_to_forward = dict(original_request.headers)
            # Optionally filter out headers you don't want to proxy
            # headers_to_forward.pop('host', None)

            # Get the HTTP client from app state
            http_client: httpx.AsyncClient = original_request.app.state.http_client
            client_timeout = httpx.Timeout(timeout or http_client.timeout)

            # Make the actual request to the target service
            resp = await http_client.request(
                method=original_request.method,
                url=target_url,
                content=request_body, # Content expects bytes
                headers=headers_to_forward,
                timeout=client_timeout
            )
            
            # Return the full response object for processing by optimize_response_async
            return resp

        # Execute the circuit-breaker-wrapped request
        try:
            upstream_response = await make_request_inner_async()
            # Track successful proxy request
            prometheus_metrics.track_proxy_request(service_name_key, "success")
            # Now we have an httpx.Response from the upstream
            logger.info(f"Response from {service_name_key} received with status {upstream_response.status_code}")
            return await optimize_response_async(upstream_response, original_request)
        except pybreaker.CircuitBreakerError as cb_error:
            # The circuit is open due to previous failures
            logger.error(f"Circuit breaker for {service_name_key} is open. Request blocked.")
            # Track circuit breaker failure
            prometheus_metrics.track_proxy_request(service_name_key, "circuit_open")
            return JSONResponse(
                status_code=503, # Service Unavailable is appropriate for circuit-breaker blocked requests
                content=GatewayErrorResponse(
                    source="api_gateway",
                    error=ErrorDetail(
                        code="CIRCUIT_BREAKER_OPEN",
                        message=f"Service '{service_name_key}' is currently unavailable due to circuit breaker being open.",
                        target_service_url=str(target_url),
                        circuit_state="open"
                    )
                ).dict()
            )
    
    # Handle specific errors from httpx client
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to {service_name_key} at {target_url}: {e}")
        # Track timeout failure
        prometheus_metrics.track_proxy_request(service_name_key, "timeout")
        return JSONResponse(
            status_code=504, # Gateway Timeout
            content=GatewayErrorResponse(
                source="api_gateway",
                error=ErrorDetail(
                    code="UPSTREAM_TIMEOUT",
                    message=f"Timeout connecting to service '{service_name_key}'.",
                    target_service_url=str(target_url),
                    original_exception=str(e)
                )
            ).dict()
        )
    except httpx.RequestError as e:
        logger.error(f"Error connecting to {service_name_key} at {target_url}: {e}")
        # Track connection failure
        prometheus_metrics.track_proxy_request(service_name_key, "connection_error")
        return JSONResponse(
            status_code=502, # Bad Gateway
            content=GatewayErrorResponse(
                source="api_gateway",
                error=ErrorDetail(
                    code="UPSTREAM_CONNECTION_ERROR",
                    message=f"Error connecting to service '{service_name_key}'.",
                    target_service_url=str(target_url),
                    original_exception=str(e)
                )
            ).dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service_name_key} at {target_url}: {e}", exc_info=True)
        # Track unexpected failure
        prometheus_metrics.track_proxy_request(service_name_key, "unexpected_error")
        return JSONResponse(
            status_code=500, # Internal Server Error
            content=GatewayErrorResponse(
                source="api_gateway",
                error=ErrorDetail(
                    code="GATEWAY_UNEXPECTED_ERROR",
                    message=f"Unexpected error in API Gateway while proxying to service '{service_name_key}'.",
                    target_service_url=str(target_url),
                    original_exception=str(e)
                )
            ).dict()
        )

# --- Routes ---
# Note: For paths with path parameters like /api/v1/model/{model_name}/config
# a more generic routing mechanism or individual routes for each would be needed.
# For now, direct mapping for existing known routes.

@app.post("/api/v1/data/fetch", tags=["Data Ingestion Proxy"], description="Proxies to Data Ingestion Service: /data/fetch")
async def data_fetch_proxy(request: Request):
    return await proxy_request_async("data_ingestion", request, path_suffix="/data/fetch", timeout=180)

@app.get("/api/v1/model/list", tags=["Model Training Proxy"], description="Proxies to Model Training Service: /model/list")
async def model_list_proxy(request: Request):
    logger.info("Handling model list request")
    try:
        response = await proxy_request_async("model_training", request, path_suffix="/model/list")
        # Add logging for debugging
        if isinstance(response, JSONResponse):
            try:
                content = response.body
                logger.info(f"Model list response: {content[:200]}")  # Log first 200 chars
            except Exception as e:
                logger.error(f"Error accessing model list response content: {e}")
        return response
    except Exception as e:
        logger.error(f"Error proxying model list request: {e}", exc_info=True)
        error_detail = ErrorDetail(code="MODEL_LIST_ERROR", message=f"Error retrieving model list: {str(e)}")
        return JSONResponse(status_code=500, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())

@app.post("/api/v1/model/train", tags=["Model Training Proxy"], description="Proxies to Model Training Service: /model/train")
async def model_train_proxy(request: Request):
    return await proxy_request_async("model_training", request, path_suffix="/model/train", timeout=3600) 

@app.post("/api/v1/model/predict", tags=["Model Training Proxy"], description="Proxies to Model Training Service: /model/predict")
async def model_predict_proxy(request: Request):
    return await proxy_request_async("model_training", request, path_suffix="/model/predict")

@app.post("/api/v1/optimize", tags=["Portfolio Optimization Proxy"], description="Proxies to Portfolio Optimization Service: /optimize")
async def optimize_portfolio_proxy(request: Request):
    return await proxy_request_async("portfolio_optimization", request, path_suffix="/optimize", timeout=300)

# --- Status Endpoints for Backend Services (using health checks) ---
async def check_backend_service_health(service_name_key: str, service_display_name: str) -> FastAPIResponse:
    service_url_base = SERVICE_URLS.get(service_name_key)
    breaker = SERVICE_BREAKERS[service_name_key]

    if not service_url_base:
        logger.warning(f"Health check for '{service_display_name}' skipped: URL not configured.")
        error_detail = ErrorDetail(code="SERVICE_NOT_CONFIGURED_GATEWAY", message=f"URL for {service_display_name} not configured.")
        # Track proxy request failure
        prometheus_metrics.track_proxy_request(service_name_key, "error")
        return JSONResponse(status_code=404, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())

    health_target_url = str(service_url_base).rstrip('/') + "/health"
    http_client: httpx.AsyncClient = app.state.http_client
    
    try:
        @breaker 
        async def perform_health_check_async(): 
            resp = await http_client.get(health_target_url, timeout=10) 
            resp.raise_for_status()
            return resp.json() 
        
        health_data = await perform_health_check_async()
        
        # Get circuit breaker state safely
        circuit_state = "closed"
        if hasattr(breaker.current_state, 'name'):
            circuit_state = breaker.current_state.name
        elif isinstance(breaker.current_state, str):
            circuit_state = breaker.current_state
        
        # Track successful proxy request
        prometheus_metrics.track_proxy_request(service_name_key, "success")
            
        return JSONResponse(content={
            "status": "success",
            "service_name": service_display_name,
            "gateway_circuit_breaker_state": circuit_state,
            "upstream_health_status": health_data
        })
    except pybreaker.CircuitBreakerError as e:
        # Get circuit breaker state safely
        circuit_state = "open"
        if hasattr(breaker.current_state, 'name'):
            circuit_state = breaker.current_state.name
        elif isinstance(breaker.current_state, str):
            circuit_state = breaker.current_state
        
        # Track circuit breaker failure
        prometheus_metrics.track_proxy_request(service_name_key, "circuit_open")
            
        logger.error(f"Circuit breaker {circuit_state} for {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code="SERVICE_CIRCUIT_OPEN", message=f"{service_display_name} is temporarily unavailable (circuit {circuit_state}).", target_service_url=health_target_url, circuit_state=circuit_state)
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    except httpx.HTTPStatusError as e:
        # Track HTTP error
        prometheus_metrics.track_proxy_request(service_name_key, f"http_error_{e.response.status_code}")
        
        logger.error(f"HTTP error from {service_display_name} health check at {health_target_url}: {e.response.status_code} - {e.response.text[:200]}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_HTTP_ERROR_{e.response.status_code}", message=f"Error from {service_display_name} health check.", target_service_url=health_target_url, original_exception=e.response.text[:200], upstream_status_code=e.response.status_code)
        return JSONResponse(status_code=e.response.status_code, content=GatewayErrorResponse(source="upstream", error=error_detail).dict())
    except httpx.RequestError as e: 
        # Track connection error
        prometheus_metrics.track_proxy_request(service_name_key, "connection_error")
        
        logger.error(f"Request error during {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_UNAVAILABLE", message=f"Could not connect to {service_display_name} health endpoint.", target_service_url=health_target_url, original_exception=str(e))
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    except json.JSONDecodeError as e:
        # Track JSON decode error
        prometheus_metrics.track_proxy_request(service_name_key, "invalid_json")
        
        logger.error(f"Failed to decode JSON from {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_INVALID_JSON", message=f"Invalid JSON response from {service_display_name} health check.", target_service_url=health_target_url, original_exception=str(e))
        return JSONResponse(status_code=500, content=GatewayErrorResponse(source="upstream", error=error_detail).dict())
    except Exception as e:
        # Track unexpected error
        prometheus_metrics.track_proxy_request(service_name_key, "unexpected_error")
        
        logger.error(f"Unexpected error during {service_display_name} health check: {e}", exc_info=True)
        error_detail = ErrorDetail(code=f"GATEWAY_HEALTH_CHECK_UNEXPECTED_ERROR", message=f"Unexpected error checking {service_display_name} health.", original_exception=str(e))
        return JSONResponse(status_code=500, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())

@app.get("/api/v1/status/data-ingestion", tags=["Service Status Proxy"], description="Checks health of Data Ingestion Service")
async def data_ingestion_status_proxy():
    return await check_backend_service_health("data_ingestion", "Data Ingestion Service")

@app.get("/api/v1/status/model-training", tags=["Service Status Proxy"], description="Checks health of Model Training Service")
async def model_training_status_proxy():
    return await check_backend_service_health("model_training", "Model Training Service")

@app.get("/api/v1/status/portfolio-optimization", tags=["Service Status Proxy"], description="Checks health of Portfolio Optimization Service")
async def portfolio_optimization_status_proxy():
    return await check_backend_service_health("portfolio_optimization", "Portfolio Optimization Service")

# Add a system-wide health check endpoint that checks all services
@app.get("/api/v1/system-status", tags=["System Status"], 
         summary="Check health of all services", 
         description="Returns health status for all microservices in the system")
async def system_status_endpoint():
    """
    Check the health status of all services in the system.
    
    Returns:
        dict: A dictionary containing the status of all services
    """
    services = ["data_ingestion", "model_training", "portfolio_optimization"]
    result = {
        "status": "success",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {},
        "overall_status": "healthy"
    }
    
    for service_name in services:
        try:
            service_url_base = SERVICE_URLS.get(service_name)
            if not service_url_base:
                result["services"][service_name] = {
                    "status": "unavailable",
                    "reason": "URL not configured"
                }
                result["overall_status"] = "degraded"
                # Track proxy request failure
                prometheus_metrics.track_proxy_request(service_name, "error")
                continue
                
            health_target_url = str(service_url_base).rstrip('/') + "/health"
            
            # Check if circuit breaker is open
            breaker = SERVICE_BREAKERS[service_name]
            circuit_state = "closed"
            if hasattr(breaker.current_state, 'name'):
                circuit_state = breaker.current_state.name
            elif isinstance(breaker.current_state, str):
                circuit_state = breaker.current_state
                
            if circuit_state == 'open':
                result["services"][service_name] = {
                    "status": "unavailable",
                    "reason": "circuit breaker open",
                    "circuit_state": circuit_state
                }
                result["overall_status"] = "degraded"
                # Track circuit breaker failure
                prometheus_metrics.track_proxy_request(service_name, "circuit_open")
                continue
                
            # Make health check request
            resp = await app.state.http_client.get(health_target_url, timeout=5.0)
            if resp.status_code == 200:
                health_data = resp.json()
                result["services"][service_name] = {
                    "status": "healthy",
                    "response_time_ms": round(resp.elapsed.total_seconds() * 1000, 2),
                    "details": health_data
                }
                # Track successful proxy request
                prometheus_metrics.track_proxy_request(service_name, "success")
            else:
                result["services"][service_name] = {
                    "status": "unhealthy",
                    "status_code": resp.status_code,
                    "reason": "Non-200 response"
                }
                result["overall_status"] = "degraded"
                # Track HTTP error
                prometheus_metrics.track_proxy_request(service_name, f"http_error_{resp.status_code}")
        except Exception as e:
            result["services"][service_name] = {
                "status": "error",
                "reason": str(e)
            }
            result["overall_status"] = "degraded"
            # Track unexpected error
            prometheus_metrics.track_proxy_request(service_name, "unexpected_error")
    
    # Check if any service is down
    if all(service["status"] == "error" or service["status"] == "unavailable" 
           for service in result["services"].values()):
        result["overall_status"] = "critical"
        
    status_code = 200
    if result["overall_status"] == "critical":
        status_code = 503  # Service Unavailable
    elif result["overall_status"] == "degraded":
        status_code = 207  # Multi-Status
        
    return JSONResponse(content=result, status_code=status_code)

# Comment out or remove old Flask app instantiation and routes for now
# app = Flask(__name__)

# @app.route('/health', methods=['GET'])
# def health_check():
# ... old health check ...

# @app.route('/api/v1/data/fetch', methods=['POST'])
# def data_fetch_proxy():
# ... etc ...