from flask import Flask, request, jsonify, Response
import requests
import os
import pybreaker
import msgpack
import time
import logging
from functools import wraps

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables to locate other services
DATA_INGESTION_SERVICE_URL = os.getenv('DATA_INGESTION_SERVICE_URL', 'http://data_ingestion_service:5001')
MODEL_TRAINING_SERVICE_URL = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://model_training_service:5002')
PORTFOLIO_OPTIMIZATION_SERVICE_URL = os.getenv('PORTFOLIO_OPTIMIZATION_SERVICE_URL', 'http://portfolio_optimization_service:5000')

# Circuit breaker configuration
circuit_breaker_config = {
    "failure_threshold": 5,  # Number of failures before opening the circuit
    "recovery_timeout": 30,  # Seconds to wait before trying again
    "retry_timeout": 60      # Seconds to wait between retries in half-open state
}

# Custom circuit breaker listeners for logging
class CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    def __init__(self, service_name):
        self.service_name = service_name

    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Circuit breaker for {self.service_name} state changed from {old_state.name} to {new_state.name}")

    def failure(self, cb, exc):
        logger.error(f"Circuit breaker for {self.service_name} recorded a failure: {exc}")

    def success(self, cb):
        logger.info(f"Circuit breaker for {self.service_name} recorded a success")

# Create circuit breakers for each service
data_ingestion_breaker = pybreaker.CircuitBreaker(
    fail_max=circuit_breaker_config["failure_threshold"],
    reset_timeout=circuit_breaker_config["recovery_timeout"],
    exclude=[requests.exceptions.ConnectionError],
    listeners=[CircuitBreakerListener("Data Ingestion Service")]
)

model_training_breaker = pybreaker.CircuitBreaker(
    fail_max=circuit_breaker_config["failure_threshold"],
    reset_timeout=circuit_breaker_config["recovery_timeout"],
    exclude=[requests.exceptions.ConnectionError],
    listeners=[CircuitBreakerListener("Model Training Service")]
)

portfolio_optimization_breaker = pybreaker.CircuitBreaker(
    fail_max=circuit_breaker_config["failure_threshold"],
    reset_timeout=circuit_breaker_config["recovery_timeout"],
    exclude=[requests.exceptions.ConnectionError],
    listeners=[CircuitBreakerListener("Portfolio Optimization Service")]
)

# Helper function to get circuit breaker state safely
def get_circuit_state(breaker):
    try:
        return breaker.current_state.name
    except AttributeError:
        return str(breaker.current_state)

@app.route('/health', methods=['GET'])
def health_check():
    # Get circuit breaker states
    circuit_status = {
        "data_ingestion": get_circuit_state(data_ingestion_breaker),
        "model_training": get_circuit_state(model_training_breaker),
        "portfolio_optimization": get_circuit_state(portfolio_optimization_breaker)
    }
    
    return jsonify({
        "status": "success",
        "message": "API Gateway is healthy and running.",
        "circuit_breaker_status": circuit_status
    }), 200

# Utility function to handle large JSON responses
def optimize_json_response(response, use_msgpack=False):
    """
    Optimizes handling of JSON responses, optionally using MessagePack for efficiency
    """
    response_headers = [(name, value) for (name, value) in response.headers.items()
                      if name.lower() not in ['content-encoding', 'transfer-encoding', 'connection', 'server', 'date']]
    
    # Check if response content type indicates JSON
    is_json_response = False
    for name, value in response.headers.items():
        if name.lower() == 'content-type' and 'application/json' in value.lower():
            is_json_response = True
            break
    
    if not is_json_response:
        # If not JSON, return raw response stream
        return Response(response.iter_content(chunk_size=32*1024), 
                        status=response.status_code, 
                        headers=response_headers)
    
    # Handle JSON response
    try:
        # Parse JSON data
        response_data = response.json()
        
        # If response is particularly large and client supports msgpack
        if use_msgpack and 'application/x-msgpack' in request.headers.get('Accept', ''):
            # Convert JSON to msgpack for more efficient transfer
            msgpack_data = msgpack.packb(response_data)
            
            # Update headers to indicate msgpack format
            response_headers.append(('Content-Type', 'application/x-msgpack'))
            return Response(msgpack_data, 
                            status=response.status_code, 
                            headers=response_headers)
        
        # Standard JSON response
        return jsonify(response_data), response.status_code, response_headers
    
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        logger.debug(f"Response content (truncated): {response.text[:1000]}")
        
        # Handle incomplete JSON response
        try:
            import json
            # Try to repair truncated JSON (simplified approach)
            text = response.text.strip()
            if text:
                # Very simple repair attempt - add closing brackets if needed
                if text.endswith(','):
                    text = text[:-1]
                
                if text.count('{') > text.count('}'):
                    text += '}' * (text.count('{') - text.count('}'))
                
                if text.count('[') > text.count(']'):
                    text += ']' * (text.count('[') - text.count(']'))
                    
                data = json.loads(text)
                logger.warning("Repaired incomplete JSON response")
                return jsonify(data), response.status_code, response_headers
        except Exception as json_err:
            logger.error(f"Failed to repair JSON: {json_err}")
        
        # Return raw response if all else fails
        return Response(response.text, status=response.status_code, headers=response_headers)

# Generic proxy function with circuit breaker integration
def proxy_request(service_url, original_request, breaker, timeout=60, use_msgpack=False):
    try:
        target_url = service_url + original_request.path.replace('/api/v1', '', 1)
        
        headers = {key: value for (key, value) in original_request.headers if key != 'Host'}
        if original_request.data and 'Content-Type' not in headers:
            headers['Content-Type'] = original_request.content_type

        logger.info(f"Proxying {original_request.method} to {target_url}")

        # Define a function to be executed with the circuit breaker
        @breaker
        def make_request():
            resp = requests.request(
                method=original_request.method,
                url=target_url,
                headers=headers,
                data=original_request.get_data(),
                params=original_request.args,
                timeout=timeout,
                stream=True  # We'll handle streaming ourselves based on content type
            )
            
            # Check if response is valid
            resp.raise_for_status()
            return resp

        # Execute request with circuit breaker
        resp = make_request()
        
        # Optimize response handling
        return optimize_json_response(resp, use_msgpack)

    except pybreaker.CircuitBreakerError as e:
        logger.error(f"Circuit breaker is open for {service_url}: {e}")
        return jsonify({
            "status": "error",
            "source": "api_gateway",
            "error": {
                "code": "SERVICE_CIRCUIT_OPEN",
                "message": f"The requested service {service_url} is temporarily unavailable due to circuit breaker.",
                "target_service_url": service_url,
                "circuit_state": breaker.current_state.name
            }
        }), 503
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to {service_url}: {e}")
        return jsonify({
            "status": "error",
            "source": "api_gateway",
            "error": {
                "code": "UPSTREAM_SERVICE_UNAVAILABLE",
                "message": f"The requested upstream service at {service_url} is unavailable.",
                "target_service_url": service_url,
                "original_exception": str(e)
            }
        }), 503
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout connecting to {service_url}: {e}")
        return jsonify({
            "status": "error",
            "source": "api_gateway",
            "error": {
                "code": "UPSTREAM_TIMEOUT",
                "message": f"The request to the upstream service at {service_url} timed out.",
                "target_service_url": service_url,
                "original_exception": str(e)
            }
        }), 504
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from {service_url}: {e}")
        # Try to extract data from error response
        try:
            error_data = e.response.json()
            return jsonify({
                "status": "error",
                "source": "upstream",
                "error": error_data,
                "upstream_status_code": e.response.status_code
            }), e.response.status_code
        except:
            return jsonify({
                "status": "error",
                "source": "upstream",
                "error": {
                    "message": str(e),
                    "upstream_status_code": e.response.status_code if hasattr(e, 'response') else "unknown"
                }
            }), e.response.status_code if hasattr(e, 'response') else 500
    except Exception as e:
        logger.error(f"Generic proxy error for {service_url}: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "source": "api_gateway",
            "error": {
                "code": "PROXY_EXECUTION_ERROR",
                "message": "An unexpected error occurred within the API Gateway while proxying the request.",
                "target_service_url": service_url,
                "original_exception": str(e)
            }
        }), 500

# --- Data Ingestion Routes ---
@app.route('/api/v1/data/fetch', methods=['POST'])
def data_fetch_proxy():
    return proxy_request(DATA_INGESTION_SERVICE_URL, request, data_ingestion_breaker, timeout=60)

# --- Model Training Routes ---
@app.route('/api/v1/model/train', methods=['POST'])
def model_train_proxy():
    # Special handling for the long-running model training process
    return proxy_request(MODEL_TRAINING_SERVICE_URL, request, model_training_breaker, timeout=900)

@app.route('/api/v1/model/predict', methods=['POST'])
def model_predict_proxy():
    return proxy_request(MODEL_TRAINING_SERVICE_URL, request, model_training_breaker, timeout=60)

# --- Portfolio Optimization Routes ---
@app.route('/api/v1/optimize', methods=['POST'])
def optimize_portfolio_proxy():
    # Use message pack for optimization requests which can have large responses
    return proxy_request(PORTFOLIO_OPTIMIZATION_SERVICE_URL, request, portfolio_optimization_breaker, timeout=180, use_msgpack=True)

# Health check endpoints for each service to test the circuit breakers
@app.route('/api/v1/status/data-ingestion', methods=['GET'])
def data_ingestion_status():
    try:
        @data_ingestion_breaker
        def check_service():
            response = requests.get(f"{DATA_INGESTION_SERVICE_URL}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        
        result = check_service()
        return jsonify({
            "status": "available",
            "circuit_state": get_circuit_state(data_ingestion_breaker),
            "service_response": result
        })
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "circuit_state": get_circuit_state(data_ingestion_breaker),
            "error": str(e)
        }), 503

@app.route('/api/v1/status/model-training', methods=['GET'])
def model_training_status():
    try:
        @model_training_breaker
        def check_service():
            response = requests.get(f"{MODEL_TRAINING_SERVICE_URL}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        
        result = check_service()
        return jsonify({
            "status": "available",
            "circuit_state": get_circuit_state(model_training_breaker),
            "service_response": result
        })
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "circuit_state": get_circuit_state(model_training_breaker),
            "error": str(e)
        }), 503

@app.route('/api/v1/status/portfolio-optimization', methods=['GET'])
def portfolio_optimization_status():
    try:
        @portfolio_optimization_breaker
        def check_service():
            response = requests.get(f"{PORTFOLIO_OPTIMIZATION_SERVICE_URL}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        
        result = check_service()
        return jsonify({
            "status": "available",
            "circuit_state": get_circuit_state(portfolio_optimization_breaker),
            "service_response": result
        })
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "circuit_state": get_circuit_state(portfolio_optimization_breaker),
            "error": str(e)
        }), 503

if __name__ == '__main__':
    # Port 5000 is specified in Dockerfile EXPOSE and CMD
    app.run(host='0.0.0.0', port=5000, debug=False) 