# Monitoring System Documentation

This document provides information about the monitoring system implemented for the Portfolio Optimizer microservices architecture.

## Overview

The monitoring system uses Prometheus for metrics collection and Grafana for visualization. Each service exposes metrics through a `/metrics` endpoint, which Prometheus scrapes at regular intervals.

## Monitored Services

The following services are monitored with consistent naming across all dashboards:

1. **API Gateway** - Main entry point for all client requests
2. **Data Ingestion** - Service for fetching and storing financial data
3. **Model Training** - Service for training ML models for prediction
4. **Portfolio Optimization** - Service for portfolio weight optimization
5. **Prometheus** (system) - The monitoring system itself
6. **Node Exporter** (system) - System metrics exporter

## Metrics Collected

### For All Services
- **HTTP Request Count** - Total number of HTTP requests processed
- **HTTP Request Duration** - Time taken to process requests
- **CPU Usage** - CPU utilization percentage
- **Memory Usage** - Memory consumption in MB
- **Service Uptime** - Time since service started

### Service-Specific Metrics

#### API Gateway
- **Circuit Breaker State** - Status of circuit breakers for downstream services
- **Proxy Request Count** - Number of requests proxied to each service

#### Model Training
- **Model Count** - Number of trained models
- **Training Time** - Time taken to train models
- **Training Loss** - Loss metrics from model training
- **Test Loss** - Validation metrics from model evaluation

#### Data Ingestion
- **Data Fetch Count** - Number of data fetch operations
- **Data Volume** - Amount of data ingested

#### Portfolio Optimization
- **Optimization Count** - Number of portfolio optimizations performed
- **Optimization Duration** - Time taken for optimization calculations

## Dashboard Panels

The Grafana dashboard consists of the following panels:

1. **Service Availability** - Shows the up/down status of all services
2. **CPU Usage** - Line graph showing CPU usage percentage for each service
3. **Memory Usage (MB)** - Line graph showing memory consumption for each service
4. **Average Response Time** - Line graph showing average request processing time
5. **Request Rate** - Line graph showing requests per second for each service
6. **Service Status** - Detailed status of each service with endpoint-specific metrics

## Accessing the Monitoring System

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (Default credentials: admin/admin)
  - Dashboard URL: http://localhost:3000/d/portfolio-dashboard-new/portfolio-optimizer-dashboard

## Configuration Files

- Prometheus configuration: `monitoring/prometheus/prometheus.yml`
- Grafana dashboards: `monitoring/grafana/provisioning/dashboards/json/portfolio-dashboard.json`

## Troubleshooting

1. Check if the service is running: `docker-compose ps`
2. Verify metrics endpoint is accessible: `curl http://localhost:<port>/metrics`
3. Check Prometheus targets: http://localhost:9090/targets
4. Review Prometheus logs: `docker-compose logs prometheus`
5. Check Grafana logs: `docker-compose logs grafana`

## Extending the Monitoring System

### Adding New Metrics

To add new metrics to a service:

1. Define the metric in the service code using the Prometheus client library
2. Expose the metric through the `/metrics` endpoint
3. Update the Grafana dashboard to visualize the new metric

### Creating Custom Dashboards

Custom dashboards can be created in Grafana UI and exported as JSON files to be stored in the repository.

## Best Practices

1. Use consistent naming across all services
2. Aggregate metrics with `sum by(job)` to avoid duplication
3. Use appropriate visualization types for different metrics
4. Set appropriate thresholds for alerts
5. Monitor both system and application-level metrics 