#!/bin/bash
set -e

# If SERVICE_MODE is set to "worker", run the worker script
if [ "$SERVICE_MODE" = "worker" ]; then
    echo "Starting in worker mode..."
    exec python worker.py
else
    echo "Starting in API server mode..."
    exec uvicorn app:app --host 0.0.0.0 --port 8000
fi 