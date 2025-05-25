FROM python:3.9-slim

# Install curl and ping for debugging network issues
RUN apt-get update && apt-get install -y curl iputils-ping && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Remove Flask-specific ENV vars
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0

# Good practice for containerized Python apps
ENV PYTHONUNBUFFERED=1 

# Port for the gateway itself
EXPOSE 5000 

# Run Uvicorn. Note: For production, consider --workers and other settings.
# The timeout here refers to keep-alive timeout. Request processing timeout is handled by httpx for upstream calls.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-keep-alive", "900"] 