# Portfolio Optimizer

A microservice-based system for optimizing investment portfolios using machine learning models with async processing capabilities.

## Architecture

The portfolio optimizer consists of the following microservices:

1. **API Gateway Service**: 
   - Routes requests to appropriate backend services
   - Provides a unified API for the frontend
   - Handles circuit breaking for resilience
   - Offers system-wide health monitoring via `/api/v1/system-status`
   - Built with FastAPI

2. **Data Ingestion Service**:
   - Fetches stock price data from financial data providers (Financial Modeling Prep)
   - Stores historical price data in PostgreSQL database
   - Built with FastAPI for high performance

3. **Model Training Service**:
   - Trains machine learning models (LSTM, GRU, Transformer) for stock price prediction
   - Manages model storage and versioning with organized directory structure
   - Provides metrics on training performance
   - Exposes predictions for portfolio optimization
   - Supports both synchronous and asynchronous training
   - Built with FastAPI

4. **Model Training Worker Service**:
   - Processes async training requests from RabbitMQ message queue
   - Enables background model training without blocking the main service
   - Handles multiple training jobs concurrently
   - Provides task status updates and notifications

5. **Portfolio Optimization Service**:
   - Uses ML model predictions to optimize portfolio weights
   - Implements modern portfolio theory algorithms (Sharpe ratio optimization)
   - Calculates key risk/return metrics
   - Integrates with model predictions for forward-looking optimization
   - Built with FastAPI

6. **Frontend Service**:
   - Provides user interface for data visualization and portfolio management
   - Enhanced progress tracking with persistent results display
   - Communicates with backend services via API Gateway
   - Built with Streamlit with improved UX

All backend services are standardized on FastAPI for consistent API documentation, improved performance, and async capabilities.

## Key Features

### Data Management
- Fetch historical stock data from Financial Modeling Prep API
- PostgreSQL database for reliable data storage
- Comprehensive data validation and error handling

### Machine Learning
- Train various neural networks (LSTM, GRU, Transformer) for time series prediction
- Model versioning and organized storage system
- Async training capabilities with worker service
- Real-time training progress tracking

### Portfolio Optimization
- Optimize portfolio weights using predicted returns and historical covariance
- Sharpe ratio maximization algorithm
- Integration with ML model predictions
- Detailed optimization metrics and breakdown

### User Interface
- Interactive web interface for data visualization and portfolio management
- Persistent training results display
- Enhanced progress tracking without auto-refresh issues
- Model selection and parameter configuration

### System Reliability
- Robust error handling and service fallback mechanisms
- Message queue integration (RabbitMQ) for async processing
- Circuit breaking for improved resilience
- Comprehensive health monitoring

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Financial Modeling Prep API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/StephanKalika/portfolio-optimizer.git
   cd portfolio-optimizer
   ```

2. Create a `.env` file with your configuration:
   ```env
   FMP_API_KEY=your_api_key_here
   DATABASE_URL=postgresql://stockuser:stockpassword@db:5432/stockdata
   RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
   ```

3. Start the services:
   ```bash
   docker compose up -d
   ```

4. Access the frontend at http://localhost:8501

## API Documentation

- API Gateway Swagger UI: http://localhost:5000/docs
- Data Ingestion Service Docs: http://localhost:5001/docs 
- Model Training Service Docs: http://localhost:5002/docs
- Portfolio Optimization Service Docs: http://localhost:5003/docs

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
- RabbitMQ Management: http://localhost:15672 (Default credentials: guest/guest)

### Monitored Services

The monitoring system tracks the following services:
- API Gateway
- Data Ingestion
- Model Training
- Model Training Worker
- Portfolio Optimization
- Frontend Service
- Database (PostgreSQL)
- Message Queue (RabbitMQ)
- Prometheus (system)
- Node Exporter (system)

## Recent Major Improvements

### Version 2.0 Features
- **Async Training System**: Added RabbitMQ-based message queue for background model training
- **Worker Service**: Dedicated worker service for processing training jobs asynchronously
- **Enhanced Progress Tracking**: Fixed frontend progress tracking with persistent results display
- **Database Query Fixes**: Resolved SQLAlchemy parameter passing issues in portfolio optimization
- **Improved UX**: Eliminated auto-refresh issues, added manual control over result display
- **Model Management**: Enhanced model storage with organized directory structure
- **Shared Modules**: Centralized common functionality across services

### Technical Improvements
- Migrated from Flask to FastAPI for enhanced performance and API documentation
- Added circuit breaking for improved resilience
- Implemented comprehensive health monitoring
- Enhanced API documentation with Swagger/ReDoc
- Upgraded data fetching and model training capabilities
- Added robust CORS support for better frontend integration
- Implemented complete monitoring dashboard with Prometheus and Grafana
- Fixed metrics collection and visualization for all services
- Standardized service naming across monitoring dashboards

### Bug Fixes
- Fixed progress tracking disappearing after training completion
- Resolved database query parameter issues in portfolio optimization
- Fixed worker service volume mounting for shared modules
- Eliminated misleading progress indicators
- Improved error handling and user feedback

## Usage Examples

### Training a Model
1. Navigate to "Model Training" in the frontend
2. Select tickers (e.g., AAPL, MSFT, GOOGL)
3. Configure model parameters (LSTM, epochs, etc.)
4. Choose training method (direct or async)
5. Monitor training progress and view results

### Portfolio Optimization
1. Navigate to "Portfolio Optimization" in the frontend
2. Select a trained model from the dropdown
3. Choose tickers for optimization
4. Set risk parameters and date ranges
5. View optimized weights and performance metrics

## Project Structure

```
portfolio-optimizer/
├── api_gateway_service/           # API Gateway service (FastAPI)
├── data_ingestion_service/        # Service for fetching stock data (FastAPI)
├── frontend_service/              # Streamlit-based UI with enhanced UX
├── model_training_service/        # Service for training prediction models (FastAPI)
├── model_training_worker/         # Async worker service for background training
├── portfolio_optimization_service/ # Service for portfolio optimization (FastAPI)
├── shared_modules/                # Common functionality across services
│   ├── database/                  # Database utilities
│   ├── message_queue/             # RabbitMQ integration
│   └── models/                    # Shared data models
├── monitoring/                    # Prometheus and Grafana configuration
│   ├── grafana/                   # Grafana dashboards and configuration
│   └── prometheus/                # Prometheus configuration
├── docker-compose.yml             # Docker Compose configuration
└── README.md                      # This file
```

## Future Enhancements

- Advanced portfolio optimization strategies (risk parity, Black-Litterman)
- Backtesting capabilities with historical performance analysis
- Sentiment analysis integration from news and social media
- Enhanced UI with interactive visualizations and charts
- Real-time data streaming and live portfolio monitoring
- Multi-asset class support (bonds, commodities, crypto)
- Risk management tools and stress testing

## Troubleshooting

### Common Issues

1. **Training Progress Disappears**: Fixed in v2.0 - results now persist until manually cleared
2. **Portfolio Optimization Errors**: Fixed database query parameter issues
3. **Worker Service Issues**: Ensure shared_modules volume is properly mounted
4. **API Gateway Timeouts**: Check service health endpoints and restart if needed

### Health Checks

Monitor service health at:
- API Gateway: http://localhost:5000/health
- Individual services: http://localhost:{port}/health

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial Modeling Prep for providing the stock market data API
- Modern Portfolio Theory for the optimization principles
- FastAPI for the excellent web framework
- Streamlit for the intuitive frontend framework
- RabbitMQ for reliable message queuing 