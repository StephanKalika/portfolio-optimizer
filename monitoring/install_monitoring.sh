#!/bin/bash

# This script installs monitoring in all services

# First create the __init__.py file
cat > monitoring/__init__.py << 'EOF'
# Prometheus monitoring package
from .prometheus_metrics import PrometheusMetrics
from .fastapi_prometheus import FastAPIPrometheusMiddleware
EOF

# Create directories if needed
mkdir -p monitoring/grafana/provisioning/dashboards/json

# Copy Prometheus monitoring files to each service
for service in api_gateway_service data_ingestion_service model_training_service portfolio_optimization_service; do
  echo "Installing monitoring in $service..."
  
  # Create monitoring directory
  mkdir -p $service/monitoring
  
  # Copy monitoring files
  cp monitoring/prometheus_metrics.py $service/monitoring/
  cp monitoring/fastapi_prometheus.py $service/monitoring/
  
  # Create __init__.py file
  cat > $service/monitoring/__init__.py << 'EOF'
# Prometheus monitoring package
from .prometheus_metrics import PrometheusMetrics
from .fastapi_prometheus import FastAPIPrometheusMiddleware
EOF

  # Add prometheus-client to requirements.txt if it doesn't exist
  if ! grep -q "prometheus-client" $service/requirements.txt; then
    echo "Adding prometheus-client to $service/requirements.txt"
    echo "prometheus-client==0.16.0" >> $service/requirements.txt
    echo "psutil==5.9.5" >> $service/requirements.txt
  fi
  
  echo "Monitoring installed in $service"
done

echo "Monitoring installation complete" 