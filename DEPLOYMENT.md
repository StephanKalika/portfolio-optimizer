# Deployment Guide for Portfolio Optimizer

This guide provides instructions for deploying the Portfolio Optimizer system in various environments.

## Docker Compose Deployment

The simplest way to deploy the system is using Docker Compose, which is suitable for both development and production environments.

### Prerequisites

- Docker and Docker Compose installed
- Git
- Financial data provider API key (e.g., Financial Modeling Prep)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/portfolio-optimizer.git
   cd portfolio-optimizer
   ```

2. Create a `.env` file with the following variables:
   ```
   FMP_API_KEY=your_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@db:5432/portfolio_optimizer
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Verify all services are running:
   ```bash
   docker-compose ps
   ```

5. Access the services:
   - Frontend: http://localhost:8501
   - API Gateway Swagger UI: http://localhost:8000/api/docs
   - Data Ingestion Service: http://localhost:5001/docs
   - Model Training Service: http://localhost:5002/docs
   - Portfolio Optimization Service: http://localhost:5003/docs

## Production Deployment Considerations

For production deployments, consider the following enhancements:

### 1. Docker Compose for Production

1. Create a production-specific Docker Compose file:
   ```bash
   touch docker-compose.prod.yml
   ```

2. Add production-specific configurations:
   ```yaml
   version: '3.8'
   
   services:
     api_gateway_service:
       restart: always
       environment:
         - LOG_LEVEL=INFO
       deploy:
         resources:
           limits:
             cpus: '0.5'
             memory: 512M
     
     # Similar configurations for other services
     
     db:
       volumes:
         - postgres_data:/var/lib/postgresql/data
       restart: always
       deploy:
         resources:
           limits:
             cpus: '1.0'
             memory: 1G
   
   volumes:
     postgres_data:
       driver: local
   ```

3. Deploy with the production configuration:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### 2. Database Considerations

For production databases:

1. Use a managed database service (AWS RDS, GCP Cloud SQL, etc.)
2. Configure proper backup schedules
3. Set up read replicas for high read workloads
4. Implement connection pooling

### 3. Security Enhancements

1. Set up HTTPS with proper certificates
2. Restrict API access with API keys
3. Configure network policies to restrict service-to-service communication
4. Use Docker secrets for sensitive information

### 4. Monitoring and Logging

1. Set up a centralized logging system (ELK stack, Loki, etc.)
2. Implement distributed tracing (Jaeger, Zipkin)
3. Configure monitoring with Prometheus and Grafana
4. Set up alerting for critical service metrics

#### Prometheus and Grafana Setup

The system includes a pre-configured monitoring stack with Prometheus and Grafana:

1. **Prometheus Configuration**:
   - Located at `monitoring/prometheus/prometheus.yml`
   - Configured to scrape metrics from all services
   - Default scrape interval: 15s

Example Prometheus configuration:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'api_gateway'
    static_configs:
      - targets: ['api_gateway_service:5000']
    metrics_path: '/metrics'
  
  - job_name: 'data_ingestion'
    static_configs:
      - targets: ['data_ingestion_service:5001']
    metrics_path: '/metrics'

  - job_name: 'model_training'
    static_configs:
      - targets: ['model_training_service:8000']
    metrics_path: '/metrics'
    
  - job_name: 'portfolio_optimization'
    static_configs:
      - targets: ['portfolio_optimization_service:8001']
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node_exporter:9100']
```

2. **Grafana Dashboard**:
   - Pre-configured dashboard available at `monitoring/grafana/provisioning/dashboards/json/portfolio-dashboard.json`
   - Dashboard URL: http://localhost:3000/d/portfolio-dashboard-new/portfolio-optimizer-dashboard
   - Default credentials: admin/admin

3. **Dashboard Features**:
   - Service Availability panel showing up/down status
   - CPU and Memory usage metrics
   - Request rates and response times
   - Service status with detailed endpoint metrics

4. **Accessing Monitoring**:
   - Prometheus UI: http://localhost:9090
   - Grafana UI: http://localhost:3000

5. **Adding Custom Metrics**:
   - Each service exposes metrics via the `/metrics` endpoint
   - Custom metrics can be added using the Prometheus client library
   - Update the Grafana dashboard to visualize new metrics

For more details, see the [MONITORING.md](MONITORING.md) documentation.

### 5. CI/CD Pipeline

1. Set up a CI/CD pipeline using GitHub Actions, GitLab CI, or Jenkins
2. Automate testing, building, and deployment
3. Implement blue-green deployments for zero-downtime updates

Example GitHub Actions workflow:
```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and push API Gateway
      uses: docker/build-push-action@v2
      with:
        context: ./api_gateway_service
        push: true
        tags: your-registry/portfolio-optimizer-api-gateway:latest
    
    # Repeat for other services
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.KEY }}
        script: |
          cd /path/to/deployment
          docker-compose pull
          docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Scaling Considerations

1. Use message queues (RabbitMQ, Kafka) for asynchronous processing
2. Implement horizontal scaling for stateless services using Docker Compose scale command:
   ```bash
   docker-compose up -d --scale model_training_service=3
   ```
3. Use Redis or other in-memory caches for frequently accessed data
4. Consider sharding for database scaling

## Backup and Disaster Recovery

1. Schedule regular database backups
2. Document and test recovery procedures
3. Implement automated backup scripts:
   ```bash
   #!/bin/bash
   # Backup script for PostgreSQL database
   
   DATE=$(date +%Y-%m-%d_%H-%M-%S)
   BACKUP_DIR="/backups"
   
   docker-compose exec -T db pg_dump -U postgres portfolio_optimizer > $BACKUP_DIR/backup_$DATE.sql
   ```

## Performance Optimization

1. Profile and optimize each service
2. Optimize database queries with proper indexing
3. Implement caching where appropriate
4. Use connection pooling for database connections 