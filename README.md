# Portfolio Optimization System

A microservice-based portfolio optimization system that uses deep learning methods for stock price prediction and portfolio optimization.

## Overview

This project implements a portfolio optimization system using:
- Deep Learning (LSTM neural networks) for stock price prediction
- Modern portfolio theory for optimization
- Microservice architecture for scalability and maintainability

## Architecture

The system is composed of the following microservices:

1. **Frontend Service**: Streamlit-based user interface
2. **API Gateway**: Routes requests to the appropriate services
3. **Data Ingestion Service**: Fetches and stores historical stock data from Financial Modeling Prep API
4. **Model Training Service**: Trains LSTM models for stock price prediction
5. **Portfolio Optimization Service**: Optimizes portfolio weights using model predictions
6. **Database**: PostgreSQL for data storage

## Key Features

- Fetch historical stock data from Financial Modeling Prep API
- Train LSTM neural networks for time series prediction
- Optimize portfolio weights using predicted returns
- Interactive web interface for data visualization and portfolio management

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Financial Modeling Prep API key (for data fetching)

### Installation

1. Clone the repository
   ```
   git clone <repository-url>
   cd portfolio-optimizer
   ```

2. Create a `.env` file in the project root with the following:
   ```
   FMP_API_KEY=your_financial_modeling_prep_api_key
   ```

3. Start the services
   ```
   docker-compose up -d
   ```

4. Access the frontend
   ```
   http://localhost:8501
   ```

## Usage

1. **Data Ingestion**: Fetch historical stock data for selected tickers
2. **Model Training**: Train LSTM models on the fetched data
3. **Portfolio Optimization**: Generate optimized portfolio weights

## Technologies Used

- **Backend**: Python, Flask, SQLAlchemy
- **Machine Learning**: PyTorch, pandas, numpy
- **Frontend**: Streamlit
- **Database**: PostgreSQL
- **Containerization**: Docker, Docker Compose
- **API Gateway**: Flask with circuit breaker pattern

## Project Structure

```
portfolio-optimizer/
├── api_gateway_service/     # API Gateway service
├── data_ingestion_service/  # Service for fetching stock data
├── frontend_service/        # Streamlit-based UI
├── model_training_service/  # Service for training prediction models
├── portfolio_optimization_service/ # Service for portfolio optimization
└── docker-compose.yml       # Docker Compose configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial Modeling Prep for providing the stock market data API
- Modern Portfolio Theory for the optimization principles 