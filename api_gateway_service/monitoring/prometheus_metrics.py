from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, multiprocess, start_http_server
import time
import threading
import psutil
import os

class PrometheusMetrics:
    def __init__(self, app_name, port=8000):
        self.app_name = app_name
        self.port = port
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_counter = Counter(
            f'{app_name}_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # Request duration metrics
        self.request_duration = Histogram(
            f'{app_name}_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status'],
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
    
    def start_metrics_server(self):
        """Start the metrics server on the specified port"""
        # Start the metrics HTTP server
        start_http_server(self.port, registry=self.registry)
        # Start system metrics collection
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
        self.request_counter.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint, status_code=status_code).observe(duration)
    
    def stop(self):
        """Stop the metrics collection thread"""
        self._running = False
        if self._metrics_thread:
            self._metrics_thread.join(timeout=1.0) 