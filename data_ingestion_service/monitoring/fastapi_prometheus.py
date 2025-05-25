import time
from fastapi import FastAPI, Request, Response
from functools import wraps
from starlette.middleware.base import BaseHTTPMiddleware
from .prometheus_metrics import PrometheusMetrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

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
        
        # Convert status_code to string for Prometheus labels with the correct names
        self.metrics.track_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
            duration=request_time
        )
        
        return response
    
    def track_function(self, name):
        """Decorator to track function execution time"""
        def decorator(f):
            @wraps(f)
            async def wrapped(*args, **kwargs):
                start_time = time.time()
                result = await f(*args, **kwargs)
                duration = time.time() - start_time
                # Use the same label names as in PrometheusMetrics class
                self.metrics.request_duration.labels(
                    method="function", 
                    path=name, 
                    code="200"
                ).observe(duration)
                return result
            return wrapped
        return decorator 