# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for numpy and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app/requirements.txt
COPY . /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
# We might need to adjust this if PyTorch installation has specific needs (e.g., CPU vs GPU, specific index URLs)
RUN pip install --no-cache-dir -r requirements.txt

# Install shared modules (will be mounted as a volume in docker-compose)
# RUN pip install -e /app/shared_modules

# Make port 8000 available (standard for Uvicorn/FastAPI)
EXPOSE 8000

# Create a directory for trained models
RUN mkdir -p /app/trained_models

# Define environment variable for unbuffered Python output
ENV PYTHONUNBUFFERED=1
# Remove Flask-specific ENV vars
# ENV FLASK_APP app.py
# ENV FLASK_RUN_HOST 0.0.0.0

# Command to run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Add worker mode entrypoint
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"] 