# Portfolio Optimizer

A microservice-based system for optimizing investment portfolios using machine learning models.

## Architecture

The portfolio optimizer consists of the following microservices:

1. **API Gateway Service**: 
   - Routes requests to appropriate backend services
   - Provides a unified API for the frontend
   - Handles circuit breaking for resilience
   - Offers system-wide health monitoring via `/api/v1/system-status`
   - Built with FastAPI

2. **Data Ingestion Service**:
   - Fetches stock price data from financial data providers
   - Stores historical price data in the database
   - Built with FastAPI for high performance

3. **Model Training Service**:
   - Trains machine learning models (LSTM, GRU, Transformer) for stock price prediction
   - Manages model storage and versioning
   - Provides metrics on training performance
   - Exposes predictions for portfolio optimization
   - Built with FastAPI

4. **Portfolio Optimization Service**:
   - Uses ML model predictions to optimize portfolio weights
   - Implements modern portfolio theory algorithms
   - Calculates key risk/return metrics
   - Built with FastAPI

5. **Frontend Service**:
   - Provides user interface for data visualization and portfolio management
   - Communicates with backend services via API Gateway
   - Built with Streamlit

All backend services are standardized on FastAPI for consistent API documentation, improved performance, and async capabilities.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Financial data provider API key (e.g., Financial Modeling Prep)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/portfolio-optimizer.git
   cd portfolio-optimizer
   ```

2. Create a `.env` file with your configuration:
   ```
   FMP_API_KEY=your_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@db:5432/portfolio_optimizer
   ```

3. Start the services:
   ```
   docker-compose up -d
   ```

4. Access the frontend at http://localhost:5000

## API Documentation

- API Gateway Swagger UI: http://localhost:5000/api/docs
- Data Ingestion Service Docs: http://localhost:5001/docs 
- Model Training Service Docs: http://localhost:8000/docs
- Portfolio Optimization Service Docs: http://localhost:8001/docs

## Monitoring

The system includes comprehensive monitoring capabilities:

- System-wide health status: GET `/api/v1/system-status`
- Individual service health: GET `/health` on each service
- Prometheus metrics: GET `/metrics` on each service
- **Prometheus and Grafana Monitoring Dashboard**: Complete system monitoring with:
  - Service Availability panel showing up/down status of all services
  - CPU Usage metrics for all microservices and system components
  - Memory Usage (MB) tracking for each service
  - Average Response Time metrics with aggregated response times
  - Request Rate tracking for all API endpoints
  - Service Status panel with detailed endpoint metrics

### Accessing Monitoring Dashboards

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (Default credentials: admin/admin)
  - Dashboard URL: http://localhost:3000/d/portfolio-dashboard-new/portfolio-optimizer-dashboard

### Monitored Services

The monitoring system tracks the following services with consistent naming:
- API Gateway
- Data Ingestion
- Model Training
- Portfolio Optimization
- Prometheus (system)
- Node Exporter (system)

## Recent Improvements

- Migrated from Flask to FastAPI for enhanced performance and API documentation
- Added circuit breaking for improved resilience
- Implemented comprehensive health monitoring
- Enhanced API documentation with Swagger/ReDoc
- Upgraded data fetching and model training capabilities
- Added robust CORS support for better frontend integration
- Implemented complete monitoring dashboard with Prometheus and Grafana
- Fixed metrics collection and visualization for all services
- Standardized service naming across monitoring dashboards

## Future Enhancements

- Message queue integration for asynchronous processing
- Advanced portfolio optimization strategies
- Backtesting capabilities
- Sentiment analysis integration
- Enhanced UI with interactive visualizations

## Key Features

- Fetch historical stock data from Financial Modeling Prep API
- Train various neural networks (LSTM, GRU, Transformer) for time series prediction
- Optimize portfolio weights using predicted returns and historical covariance
- Interactive web interface for data visualization and portfolio management
- Robust error handling and service fallback mechanisms

## Project Structure

```
portfolio-optimizer/
├── api_gateway_service/     # API Gateway service (FastAPI)
├── data_ingestion_service/  # Service for fetching stock data (FastAPI)
├── frontend_service/        # Streamlit-based UI
├── model_training_service/  # Service for training prediction models (FastAPI)
├── portfolio_optimization_service/ # Service for portfolio optimization (FastAPI)
├── monitoring/              # Prometheus and Grafana configuration
│   ├── grafana/             # Grafana dashboards and configuration
│   └── prometheus/          # Prometheus configuration
└── docker-compose.yml       # Docker Compose configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial Modeling Prep for providing the stock market data API
- Modern Portfolio Theory for the optimization principles 