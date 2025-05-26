# Portfolio Optimizer v2.0.0 Deployment Guide

This guide provides step-by-step instructions for deploying the Portfolio Optimizer system.

## Prerequisites

### System Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space
- Internet connection for data fetching

### API Keys Required
- Financial Modeling Prep API key (free tier available)
  - Sign up at: https://financialmodelingprep.com/developer/docs
  - Free tier includes 250 requests/day

## Quick Start Deployment

### 1. Clone Repository
```bash
git clone https://github.com/StephanKalika/portfolio-optimizer.git
cd portfolio-optimizer
```

### 2. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your API key
nano .env  # or use your preferred editor
```

**Required Configuration:**
```env
FMP_API_KEY=your_actual_api_key_here
```

### 3. Deploy Services
```bash
# Start all services
docker compose up -d

# Verify deployment
docker compose ps
```

### 4. Access Applications
- **Frontend**: http://localhost:8501
- **API Gateway**: http://localhost:5000/docs
- **Monitoring (Grafana)**: http://localhost:3000 (admin/admin)
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)

## Service Architecture

### Core Services
| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 8501 | Streamlit web interface |
| API Gateway | 5000 | Request routing and circuit breaking |
| Data Ingestion | 5001 | Stock data fetching |
| Model Training | 5002 | ML model training |
| Portfolio Optimization | 5003 | Portfolio weight optimization |
| Model Training Worker | - | Async training processing |

### Infrastructure Services
| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Data storage |
| RabbitMQ | 5672/15672 | Message queue |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboard |

## Health Checks

### System Status
```bash
# Check all services
curl http://localhost:5000/api/v1/system-status

# Individual service health
curl http://localhost:5000/health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

### Service Logs
```bash
# View all logs
docker compose logs

# Specific service logs
docker compose logs frontend_service
docker compose logs model_training_service
docker compose logs portfolio_optimization_service
```

## Usage Workflow

### 1. Data Ingestion
1. Navigate to "Data Ingestion" in the frontend
2. Enter stock tickers (e.g., AAPL,MSFT,GOOGL)
3. Select date range (recommend 3+ years)
4. Click "Fetch Data"

### 2. Model Training
1. Navigate to "Model Training"
2. Select tickers to train
3. Configure model parameters:
   - Model type: LSTM (recommended)
   - Epochs: 50 (default)
   - Sequence length: 60 (default)
4. Choose training method:
   - **Direct**: Immediate training with progress display
   - **Async**: Background training via worker service
5. Monitor training progress and results

### 3. Portfolio Optimization
1. Navigate to "Portfolio Optimization"
2. Select trained model from dropdown
3. Choose tickers for portfolio
4. Set optimization parameters:
   - Risk-free rate: 0.02 (2%)
   - Date range for covariance calculation
5. Click "Optimize Portfolio"
6. View optimized weights and metrics

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker status
docker compose ps

# Restart specific service
docker compose restart service_name

# Full system restart
docker compose down && docker compose up -d
```

#### Database Connection Issues
```bash
# Check database logs
docker compose logs db

# Reset database
docker compose down -v
docker compose up -d
```

#### Training Progress Issues
- **Fixed in v2.0**: Progress now persists until manually cleared
- Use manual refresh buttons instead of waiting for auto-refresh
- Check worker service logs if using async training

#### Portfolio Optimization Errors
- **Fixed in v2.0**: Database query parameter issues resolved
- Ensure model is fully trained before optimization
- Verify tickers exist in selected model

### Performance Optimization

#### Memory Usage
```bash
# Monitor resource usage
docker stats

# Adjust service resources in docker-compose.yml if needed
```

#### Training Performance
- Use async training for multiple models
- Monitor worker service capacity
- Consider reducing epochs for faster training

## Monitoring and Maintenance

### Grafana Dashboard
1. Access: http://localhost:3000
2. Login: admin/admin
3. Navigate to Portfolio Optimizer Dashboard
4. Monitor:
   - Service availability
   - CPU and memory usage
   - Request rates and response times

### RabbitMQ Management
1. Access: http://localhost:15672
2. Login: guest/guest
3. Monitor:
   - Queue status
   - Message rates
   - Worker connections

### Log Management
```bash
# Rotate logs
docker compose logs --tail=100 service_name

# Clear old containers
docker system prune
```

## Backup and Recovery

### Data Backup
```bash
# Backup database
docker compose exec db pg_dump -U stockuser stockdata > backup.sql

# Backup trained models
docker compose exec model_training_service tar -czf models_backup.tar.gz /app/trained_models
```

### Recovery
```bash
# Restore database
docker compose exec -T db psql -U stockuser stockdata < backup.sql

# Restore models
docker compose exec model_training_service tar -xzf models_backup.tar.gz -C /app/
```

## Security Considerations

### Production Deployment
1. **Change default passwords**:
   - Database credentials
   - RabbitMQ credentials
   - Grafana admin password

2. **Network Security**:
   - Use reverse proxy (nginx/traefik)
   - Enable HTTPS/TLS
   - Restrict port access

3. **API Security**:
   - Implement authentication
   - Rate limiting
   - Input validation

### Environment Variables
```env
# Production example
POSTGRES_PASSWORD=secure_random_password
RABBITMQ_DEFAULT_PASS=secure_random_password
GRAFANA_ADMIN_PASSWORD=secure_random_password
```

## Scaling Considerations

### Horizontal Scaling
- Multiple worker instances for training
- Load balancer for API Gateway
- Database read replicas

### Vertical Scaling
- Increase container memory limits
- Add CPU cores for training services
- SSD storage for database

## Support

### Documentation
- API Documentation: Available at service `/docs` endpoints
- Monitoring: Grafana dashboards with detailed metrics
- Logs: Comprehensive logging across all services

### Community
- GitHub Issues: Report bugs and feature requests
- Discussions: Architecture and usage questions

## Version Information
- **Current Version**: 2.0.0
- **Release Date**: 2025-05-26
- **Compatibility**: Docker Compose 2.0+, Docker Engine 20.10+ 