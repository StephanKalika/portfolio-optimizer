# Monitoring System for Portfolio Optimizer

This directory contains the monitoring system for the Portfolio Optimizer microservices architecture using Prometheus and Grafana.

## Components

1. **Prometheus**: Collects and stores metrics from all services
2. **Grafana**: Visualizes metrics in customizable dashboards
3. **Node Exporter**: Collects system metrics from the host machine
4. **Service Instrumentation**: Python libraries for adding metrics to services

## Setup

The monitoring system is included in the main `docker-compose.yml` file and will start automatically with the rest of the services.

### Manual Setup

If you want to run the monitoring system separately:

```bash
cd monitoring
docker-compose up -d
```

### Adding Monitoring to Services

To add monitoring to your services, run the installation script:

```bash
./monitoring/install_monitoring.sh
```

This script will:
1. Copy the necessary monitoring files to each service
2. Add the required dependencies to each service's requirements.txt
3. Create the necessary directory structure

## Integration in Services

### FastAPI Applications

```python
from monitoring import FastAPIPrometheusMiddleware

app = FastAPI()
app.add_middleware(FastAPIPrometheusMiddleware, app_name="my_service", metrics_port=9100)

# Use the track_function decorator for specific functions
@app.get("/something")
@metrics.track_function("important_calculation")
async def important_calculation():
    # Your code here
    pass
```

## Available Dashboards

1. **Portfolio Optimizer Dashboard**: Main dashboard showing service availability, CPU/memory usage, and response times
2. **System Overview**: Shows system-level metrics from node-exporter

## Accessing the Dashboards

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Adding Custom Metrics

You can add custom metrics to your services by accessing the metrics object:

```python
# Track a specific event
metrics.metrics.request_counter.labels(method="custom", endpoint="batch_job", status_code=200).inc()

# Track execution time
start_time = time.time()
result = do_something()
duration = time.time() - start_time
metrics.metrics.request_duration.labels(method="custom", endpoint="batch_job", status_code=200).observe(duration)
``` 