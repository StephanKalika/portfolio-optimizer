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

# Configure logging (FastAPI uses uvicorn logging by default, but good to have a logger instance)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="API Gateway Service", version="1.0.0")

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
    
    service_url_base = SERVICE_URLS.get(service_name_key)
    if not service_url_base:
        logger.error(f"Service URL for '{service_name_key}' is not configured in API Gateway.")
        # Return Pydantic model based error response
        error_detail = ErrorDetail(code="SERVICE_NOT_CONFIGURED_GATEWAY", message=f"Service '{service_name_key}' not configured.")
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())

    target_url = str(service_url_base).rstrip('/') + path_suffix
    breaker = SERVICE_BREAKERS[service_name_key]
    http_client: httpx.AsyncClient = app.state.http_client

    headers = {key: value for key, value in original_request.headers.items() if key.lower() != 'host'}
    body_bytes = await original_request.body()
    
    # Preserve original Content-Type if present, otherwise httpx might set its own based on content
    if body_bytes and 'content-type' not in headers and original_request.headers.get('content-type'):
         headers['content-type'] = original_request.headers['content-type']
    elif body_bytes and 'content-type' not in headers: # If no content-type but body exists, default to json
        headers['content-type'] = 'application/json'

    logger.info(f"Proxying {original_request.method} to {target_url} for service '{service_name_key}' with timeout {timeout or http_client.timeout}")

    try:
        @breaker
        async def make_request_inner_async(): 
            response = await http_client.request(
                method=original_request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                params=original_request.query_params, # httpx.QueryParams or Dict
                timeout=timeout # Uses specific timeout if provided, else client default
            )
            response.raise_for_status() 
            return response

        upstream_response = await make_request_inner_async()
        return await optimize_response_async(upstream_response, original_request)

    except pybreaker.CircuitBreakerError as e:
        logger.error(f"Circuit breaker is OPEN for {service_name_key} at {target_url}: {e}")
        error_detail = ErrorDetail(code="SERVICE_CIRCUIT_OPEN", message=f"Service {service_name_key} is temporarily unavailable (circuit open).", target_service_url=target_url, circuit_state=breaker.current_state.name)
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    
    except httpx.ConnectError as e:
        logger.error(f"Connection error to {target_url} ({service_name_key}): {e}")
        error_detail = ErrorDetail(code="UPSTREAM_CONNECTION_ERROR", message=f"Cannot connect to {service_name_key}.", target_service_url=target_url, original_exception=str(e))
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting/requesting {target_url} ({service_name_key}): {e}")
        error_detail = ErrorDetail(code="UPSTREAM_TIMEOUT", message=f"Request to {service_name_key} timed out.", target_service_url=target_url, original_exception=str(e))
        return JSONResponse(status_code=504, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    
    except httpx.HTTPStatusError as e: 
        logger.error(f"HTTP error {e.response.status_code} from {target_url} ({service_name_key}). Response: {e.response.text[:500]}")
        try:
            error_content = e.response.json()
            if isinstance(error_content, dict) and "error" in error_content and isinstance(error_content["error"], dict):
                upstream_error_detail_dict = error_content["error"]
                if "code" not in upstream_error_detail_dict or "message" not in upstream_error_detail_dict:
                     upstream_error_detail_dict = {"code": f"UPSTREAM_ERROR_{e.response.status_code}", "message": json.dumps(error_content) }
            elif isinstance(error_content, dict) and "detail" in error_content: 
                upstream_error_detail_dict = {"code": f"UPSTREAM_ERROR_{e.response.status_code}", "message": str(error_content["detail"])}
            else:
                upstream_error_detail_dict = {"code": f"UPSTREAM_ERROR_{e.response.status_code}", "message": e.response.text[:200]}
            
            # Ensure all required fields for ErrorDetail are present or defaulted
            final_error_detail = ErrorDetail(
                code=upstream_error_detail_dict.get("code", f"UPSTREAM_ERROR_{e.response.status_code}"),
                message=upstream_error_detail_dict.get("message", "Unknown upstream error."),
                target_service_url=target_url,
                upstream_status_code=e.response.status_code
            )

        except json.JSONDecodeError:
            final_error_detail = ErrorDetail(code=f"UPSTREAM_ERROR_{e.response.status_code}", message=e.response.text[:200], target_service_url=target_url, upstream_status_code=e.response.status_code)
        
        return JSONResponse(status_code=e.response.status_code, content=GatewayErrorResponse(source="upstream", error=final_error_detail).dict())
    
    except Exception as e:
        logger.error(f"Generic error proxying to {target_url} ({service_name_key}): {type(e).__name__} - {e}", exc_info=True)
        error_detail = ErrorDetail(code="GATEWAY_PROXY_ERROR", message="An unexpected error occurred in the API Gateway while proxying.", original_exception=str(e))
        return JSONResponse(status_code=500, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())

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
        return JSONResponse(content={
            "status": "success",
            "service_name": service_display_name,
            "gateway_circuit_breaker_state": breaker.current_state.name,
            "upstream_health_status": health_data
        })
    except pybreaker.CircuitBreakerError as e:
        logger.error(f"Circuit breaker OPEN for {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code="SERVICE_CIRCUIT_OPEN", message=f"{service_display_name} is temporarily unavailable (circuit open).", target_service_url=health_target_url, circuit_state=breaker.current_state.name)
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {service_display_name} health check at {health_target_url}: {e.response.status_code} - {e.response.text[:200]}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_HTTP_ERROR_{e.response.status_code}", message=f"Error from {service_display_name} health check.", target_service_url=health_target_url, original_exception=e.response.text[:200], upstream_status_code=e.response.status_code)
        return JSONResponse(status_code=e.response.status_code, content=GatewayErrorResponse(source="upstream", error=error_detail).dict())
    except httpx.RequestError as e: 
        logger.error(f"Request error during {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_UNAVAILABLE", message=f"Could not connect to {service_display_name} health endpoint.", target_service_url=health_target_url, original_exception=str(e))
        return JSONResponse(status_code=503, content=GatewayErrorResponse(source="api_gateway", error=error_detail).dict())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {service_display_name} health check at {health_target_url}: {e}")
        error_detail = ErrorDetail(code=f"UPSTREAM_HEALTH_INVALID_JSON", message=f"Invalid JSON response from {service_display_name} health check.", target_service_url=health_target_url, original_exception=str(e))
        return JSONResponse(status_code=500, content=GatewayErrorResponse(source="upstream", error=error_detail).dict())
    except Exception as e:
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

# Comment out or remove old Flask app instantiation and routes for now
# app = Flask(__name__)

# @app.route('/health', methods=['GET'])
# def health_check():
# ... old health check ...

# @app.route('/api/v1/data/fetch', methods=['POST'])
# def data_fetch_proxy():
# ... etc ...