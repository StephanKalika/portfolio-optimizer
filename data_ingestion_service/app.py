from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import requests # For making HTTP requests to FMP
import time
import os # Import os to access environment variables
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json
import logging
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Data Ingestion Service")

# Get API key from environment variable
FMP_API_KEY = os.getenv('FMP_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

# Setup database engine
db_engine = None

# Define models for request and response
class DataFetchRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v:
            raise ValueError("Tickers list cannot be empty")
        if not all(isinstance(t, str) for t in v):
            raise ValueError("All items in 'tickers' list must be strings")
        return v

class TickerResult(BaseModel):
    ticker: str
    status: str = "pending"
    rows_upserted: int = 0
    details: Union[str, Dict[str, Any]] = ""

class DataFetchResponse(BaseModel):
    status: str
    message: str
    total_rows_upserted: int
    processed_tickers: List[TickerResult]

def initialize_db_engine(retries=5, delay=5):
    global db_engine
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set. Database functionality will be unavailable.")
        return

    for attempt in range(retries):
        try:
            current_engine = create_engine(DATABASE_URL)
            # Test connection and create table if it doesn't exist
            with current_engine.connect() as connection:
                connection.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    adj_close NUMERIC(10, 4) NOT NULL,
                    UNIQUE(date, ticker)
                );
                """))
            db_engine = current_engine # Assign to global only on success
            logger.info(f"Database engine created and stock_prices table ensured after {attempt + 1} attempts.")
            return
        except SQLAlchemyError as e:
            logger.error(f"Error creating database engine or table (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Database engine initialization failed.")
                db_engine = None # Ensure db_engine is None if setup fails

# Call initialization at startup
@app.on_event("startup")
async def startup_event():
    initialize_db_engine()

@app.get('/health')
async def health_check():
    """
    Health check endpoint.
    """
    db_status_message = "disconnected"
    api_key_ok = bool(FMP_API_KEY)
    db_ok = False

    if db_engine:
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            db_status_message = "connected"
            db_ok = True
        except SQLAlchemyError:
            db_status_message = "connection_error"
            logger.error("Health check: Database connection error.", exc_info=True)
    else:
        db_status_message = "not_initialized"

    if not api_key_ok:
        logger.warning("FMP_API_KEY environment variable is not set.")
        
    if db_ok: # Consider API key presence for full health if it's critical
        payload = {
            "status": "success",
            "data": {
                "service_status": "healthy",
                "api_key_set": api_key_ok,
                "database_status": db_status_message
            },
            "message": "Data ingestion service is healthy."
        }
        return JSONResponse(content=payload, status_code=200)
    else:
        payload = {
            "status": "error",
            "error": {
                "code": "SERVICE_UNHEALTHY",
                "message": "Data ingestion service is unhealthy or has configuration issues."
            },
            "details": {
                 "api_key_set": api_key_ok, # Still report API key status
                 "database_status": db_status_message
            }
        }
        return JSONResponse(content=payload, status_code=503)

@app.post('/data/fetch', response_model=DataFetchResponse)
async def fetch_data(request: DataFetchRequest):
    """
    Endpoint to trigger data fetching for specified tickers and date range using Financial Modeling Prep.
    """
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not configured, cannot fetch data.")
        payload = {
            "status": "error", 
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": "API key for FMP (Financial Modeling Prep) is not configured."
            },
            "details": "Service cannot fetch data without FMP_API_KEY."
        }
        raise HTTPException(status_code=503, detail=payload)
        
    if not db_engine:
        logger.error("Database is not available.")
        payload = {
            "status": "error",
            "error": {
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database connection is not available."
            },
            "details": "Service cannot store data without a database connection."
        }
        raise HTTPException(status_code=503, detail=payload)
    
    tickers = request.tickers
    start_date_str = request.start_date
    end_date_str = request.end_date

    processed_tickers_summary = []
    total_rows_upserted_overall = 0

    for i, ticker_symbol in enumerate(tickers):
        logger.info(f"Processing ticker: {ticker_symbol} [{i+1}/{len(tickers)}]")
        ticker_result = TickerResult(ticker=ticker_symbol)
        try:
            url = f"{FMP_BASE_URL}/historical-price-full/{ticker_symbol}?from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status()
            fmp_data = response.json()

            if not fmp_data or 'historical' not in fmp_data or not isinstance(fmp_data['historical'], list) or not fmp_data['historical']:
                logger.warning(f"No valid historical data from FMP for {ticker_symbol}. Response: {str(fmp_data)[:200]}...")
                ticker_result.status = "no_data_from_api"
                ticker_result.details = "No historical data found or data in unexpected format from FMP."
                processed_tickers_summary.append(ticker_result)
                if i < len(tickers) - 1: time.sleep(1)
                continue
                
            df = pd.DataFrame(fmp_data['historical'])
            if df.empty or 'adjClose' not in df.columns or 'date' not in df.columns:
                logger.warning(f"Historical data for {ticker_symbol} (FMP) parsed to empty or malformed DataFrame. Columns: {df.columns}")
                ticker_result.status = "empty_or_malformed_data"
                ticker_result.details = "API data parsed to empty or malformed DataFrame."
                processed_tickers_summary.append(ticker_result)
                if i < len(tickers) - 1: time.sleep(1)
                continue

            df = df[['date', 'adjClose']].copy()
            df.rename(columns={'adjClose': 'adj_close'}, inplace=True)
            df['ticker'] = ticker_symbol
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')
            df.dropna(subset=['date', 'adj_close'], inplace=True)

            if df.empty:
                logger.warning(f"Data for {ticker_symbol} became empty after cleaning. Skipping.")
                ticker_result.status = "empty_after_cleaning"
                ticker_result.details = "Data became empty after type conversion or NaN removal."
                processed_tickers_summary.append(ticker_result)
                if i < len(tickers) - 1: time.sleep(1)
                continue

            upsert_sql = text("""
            INSERT INTO stock_prices (date, ticker, adj_close)
            VALUES (:date, :ticker, :adj_close)
            ON CONFLICT (date, ticker) DO UPDATE SET adj_close = EXCLUDED.adj_close;
            """)
            
            rows_for_upsert = df.to_dict(orient='records')
            logger.info(f"Attempting to upsert {len(rows_for_upsert)} rows for {ticker_symbol}.")
            try:
                with db_engine.connect() as connection:
                    with connection.begin(): # Start a transaction
                        connection.execute(upsert_sql, rows_for_upsert) 
                current_ticker_rows = len(rows_for_upsert)
                total_rows_upserted_overall += current_ticker_rows
                ticker_result.status = "success"
                ticker_result.rows_upserted = current_ticker_rows
                ticker_result.details = f"Successfully processed and initiated upsert for {current_ticker_rows} rows."
                logger.info(f"Successfully initiated upsert for {current_ticker_rows} rows for {ticker_symbol}.")
            except SQLAlchemyError as db_err:
                logger.error(f"Database error during upsert for {ticker_symbol}: {db_err}", exc_info=True)
                ticker_result.status = "db_upsert_error"
                ticker_result.details = f"Database error: {str(db_err)}"
            except Exception as general_db_err:
                logger.error(f"Unexpected error during DB upsert for {ticker_symbol}: {general_db_err}", exc_info=True)
                ticker_result.status = "unexpected_db_error"
                ticker_result.details = f"Unexpected DB error: {str(general_db_err)}"
            
            processed_tickers_summary.append(ticker_result)

            if i < len(tickers) - 1: time.sleep(1)

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else 'N/A'
            err_code = "API_CLIENT_ERROR" if isinstance(status_code, int) and 400 <= status_code < 500 else "API_SERVER_ERROR"
            log_msg = f"HTTP error for {ticker_symbol} (FMP): {http_err}. Status: {status_code}. Response: {http_err.response.text[:200] if http_err.response else 'No response text'}..."
            logger.error(log_msg)
            ticker_result.status = "api_http_error"
            ticker_result.details = {"code": err_code, "message": str(http_err), "fmp_status_code": status_code}
            processed_tickers_summary.append(ticker_result)
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error (e.g., connection, timeout) for {ticker_symbol} (FMP): {req_err}", exc_info=True)
            ticker_result.status = "api_request_error"
            ticker_result.details = {"code": "API_CONNECTION_ERROR", "message": str(req_err)}
            processed_tickers_summary.append(ticker_result)
        except Exception as e: # Catch-all for unexpected errors during a ticker's processing
            logger.error(f"Unexpected error processing {ticker_symbol}: {e}", exc_info=True)
            ticker_result.status = "error"
            ticker_result.details = str(e)
            processed_tickers_summary.append(ticker_result)
        
    final_message = f"Data fetching process completed. Total rows upserted attempt count: {total_rows_upserted_overall}."
    if total_rows_upserted_overall == 0 and any(t.status != 'success' for t in processed_tickers_summary):
        final_message = "Data fetching process completed, but no new data was successfully processed or upserted."
        
    # Construct the response data
    response_payload = {
        "status": "completed", 
        "message": f"Data fetching process completed for {len(tickers)} requested tickers.",
        "total_rows_upserted": total_rows_upserted_overall,
        "processed_tickers": processed_tickers_summary
    }
    
    return response_payload

@app.post('/admin/shutdown')
async def shutdown():
    """
    Endpoint to gracefully shut down the service.
    """
    logger.info("Shutting down the service.")
    # Implement the logic to gracefully shut down the service
    payload = {
        "status": "success",
        "message": "Service is shutting down."
    }
    return JSONResponse(content=payload, status_code=200)

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=True) 