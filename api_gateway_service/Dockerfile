FROM python:3.9-slim

# Install curl, ping, and build dependencies for psutil
RUN apt-get update && apt-get install -y curl iputils-ping gcc python3-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install aio-pika
RUN pip install aio-pika

# Copy shared modules (don't install as editable, just use them directly)
COPY ./shared_modules /app/shared_modules

COPY . /app

# Good practice for containerized Python apps
ENV PYTHONUNBUFFERED=1 
ENV PYTHONPATH=/app

# Port for the gateway itself
EXPOSE 5000 

# Run Uvicorn. Note: For production, consider --workers and other settings.
# The timeout here refers to keep-alive timeout. Request processing timeout is handled by httpx for upstream calls.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-keep-alive", "900"] 