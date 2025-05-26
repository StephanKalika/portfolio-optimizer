# Changelog

All notable changes to the Portfolio Optimizer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-05-26

### Added
- **Async Training System**: Implemented RabbitMQ-based message queue for background model training
- **Model Training Worker Service**: Dedicated worker service for processing training jobs asynchronously
- **Shared Modules**: Centralized common functionality across services
  - Database utilities in `shared_modules/database/`
  - Message queue integration in `shared_modules/message_queue/`
  - Shared data models in `shared_modules/models/`
- **Enhanced Progress Tracking**: Persistent training results display in frontend
- **Manual Control**: Added manual refresh buttons instead of automatic page refresh
- **RabbitMQ Integration**: Message queue for reliable async task processing
- **Worker Health Monitoring**: Health checks for worker service status
- **Detailed Training Results**: Per-ticker training metrics and performance data
- **Model Management**: Organized model storage with versioned directory structure

### Fixed
- **Progress Tracking Issues**: Resolved disappearing progress information after training completion
- **Database Query Parameters**: Fixed SQLAlchemy parameter passing in portfolio optimization service
- **Worker Service Volume Mounting**: Added missing shared_modules volume mount
- **Misleading Progress Indicators**: Eliminated incorrect "100% with 0/0 epochs" displays
- **Auto-Refresh Problems**: Removed automatic page refresh that cleared results
- **Session State Management**: Improved persistent storage of training results
- **Error Handling**: Enhanced error messages and user feedback

### Changed
- **Frontend UX**: Replaced complex progress tracking with simple spinner approach
- **Training Endpoints**: Added both direct and async training options
- **Result Display**: Training results now persist until manually cleared
- **Progress Monitoring**: Switched from misleading progress bars to reliable status indicators
- **Service Architecture**: Enhanced microservice communication with message queuing

### Improved
- **User Experience**: No more disappearing results, better control over interface
- **System Reliability**: Async processing prevents blocking operations
- **Error Feedback**: Clear success/failure messages with detailed information
- **Performance**: Background training doesn't block the main service
- **Monitoring**: Enhanced service health checks and status reporting

## [1.5.0] - 2025-05-25

### Added
- **Comprehensive Monitoring**: Prometheus and Grafana integration
- **Service Health Checks**: Individual and system-wide health monitoring
- **Metrics Collection**: CPU, memory, and request metrics for all services
- **Grafana Dashboard**: Visual monitoring dashboard with service status panels

### Fixed
- **API Gateway Circuit Breaking**: Improved service resilience
- **Metrics Endpoint**: Standardized `/metrics` endpoint across all services
- **Service Discovery**: Enhanced service communication reliability

## [1.0.0] - 2025-05-20

### Added
- **Initial Release**: Complete portfolio optimization system
- **Microservice Architecture**: API Gateway, Data Ingestion, Model Training, Portfolio Optimization, Frontend
- **FastAPI Migration**: Migrated all backend services from Flask to FastAPI
- **Machine Learning Models**: LSTM, GRU, and Transformer support for time series prediction
- **Portfolio Optimization**: Sharpe ratio maximization algorithm
- **Data Integration**: Financial Modeling Prep API integration
- **Frontend Interface**: Streamlit-based user interface
- **Docker Containerization**: Complete Docker Compose setup
- **Database Integration**: PostgreSQL for data storage
- **API Documentation**: Swagger/ReDoc documentation for all services

### Features
- Historical stock data fetching and storage
- Neural network model training for price prediction
- Portfolio weight optimization using model predictions
- Interactive web interface for data visualization
- Comprehensive error handling and logging
- CORS support for frontend integration
- Robust service architecture with health monitoring

## Technical Debt and Future Improvements

### Planned for v2.1.0
- **Advanced Optimization Strategies**: Risk parity, Black-Litterman model
- **Backtesting Framework**: Historical performance analysis
- **Real-time Data Streaming**: Live market data integration
- **Enhanced UI**: Interactive charts and visualizations
- **Multi-asset Support**: Bonds, commodities, cryptocurrency

### Known Issues
- API Gateway SERVICE_BREAKERS error (minor, doesn't affect core functionality)
- Docker API version compatibility on some systems
- Occasional timeout issues with large training datasets

## Migration Guide

### From v1.x to v2.0
1. **Update Docker Compose**: New services added (worker, rabbitmq)
2. **Environment Variables**: Add RABBITMQ_URL to .env file
3. **Volume Mounts**: Ensure shared_modules volume is properly configured
4. **Database**: No schema changes required
5. **Frontend**: New features available, existing functionality preserved

### Breaking Changes
- None - v2.0 is fully backward compatible with v1.x APIs
- New async training endpoints are additive, not replacing existing ones

## Contributors
- Stephan Kalika - Lead Developer
- Portfolio Optimizer Team

## Acknowledgments
- Financial Modeling Prep for market data API
- FastAPI community for excellent framework
- Streamlit team for intuitive frontend framework
- RabbitMQ for reliable message queuing 