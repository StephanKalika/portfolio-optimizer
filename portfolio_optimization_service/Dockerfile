# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8001 available
EXPOSE 8001

# Define environment variable for unbuffered Python output
ENV PYTHONUNBUFFERED=1

# Run Uvicorn server
CMD echo "DEBUG: Starting portfolio_optimization_service with latest Dockerfile CMD" && uvicorn app:app --host 0.0.0.0 --port 8001 