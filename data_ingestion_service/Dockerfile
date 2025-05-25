# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Gunicorn will be installed here if added to requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Remove Flask-specific ENV vars for running dev server
# ENV FLASK_APP app.py
# ENV FLASK_RUN_HOST 0.0.0.0

# Run the FastAPI app with uvicorn when the container launches
# Using workers=4 and timeout=120s for potentially long FMP API calls / DB operations
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "4", "--timeout-keep-alive", "120"] 