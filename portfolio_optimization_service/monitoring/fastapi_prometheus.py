import time
from fastapi import FastAPI, Request, Response
from functools import wraps
from starlette.middleware.base import BaseHTTPMiddleware
from .prometheus_metrics import PrometheusMetrics

class FastAPIPrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, app_name: str = "fastapi_app", metrics_port: int = 9100):
        super().__init__(app)
        self.app = app
        self.app_name = app_name
        self.metrics = PrometheusMetrics(app_name, metrics_port)
        
        # Start the metrics server
        self.metrics.start_metrics_server()
        
        # Add metrics endpoint
        @app.get("/metrics")
        async def metrics():
            return Response(
                content=f"Metrics available on port {metrics_port}",
                media_type="text/plain"
            )
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        
        # Skip tracking metrics requests
        if request.url.path != "/metrics":
            request_time = time.time() - start_time
            self.metrics.track_request(
                request.method,
                request.url.path,
                response.status_code,
                request_time
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
                self.metrics.request_duration.labels(
                    method="function", 
                    endpoint=name, 
                    status_code=200
                ).observe(duration)
                return result
            return wrapped
        return decorator 