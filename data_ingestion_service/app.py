from flask import Flask, jsonify, request
import pandas as pd
import requests # For making HTTP requests to FMP
import time
import os # Import os to access environment variables
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

app = Flask(__name__)

# Get API key from environment variable
FMP_API_KEY = os.getenv('FMP_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

# Setup database engine
db_engine = None

def initialize_db_engine(retries=5, delay=5):
    global db_engine
    if not DATABASE_URL:
        app.logger.error("DATABASE_URL environment variable is not set. Database functionality will be unavailable.")
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
            app.logger.info(f"Database engine created and stock_prices table ensured after {attempt + 1} attempts.")
            return
        except SQLAlchemyError as e:
            app.logger.error(f"Error creating database engine or table (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                app.logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                app.logger.error("Max retries reached. Database engine initialization failed.")
                db_engine = None # Ensure db_engine is None if setup fails

initialize_db_engine() # Call initialization at startup

@app.route('/health', methods=['GET'])
def health_check():
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
            app.logger.error("Health check: Database connection error.", exc_info=True)
    else:
        db_status_message = "not_initialized"

    if not api_key_ok:
        app.logger.warning("FMP_API_KEY environment variable is not set.")
        
    if db_ok: # Consider API key presence for full health if it's critical
        return jsonify({
                "status": "success",
                "data": {
                    "service_status": "healthy",
                    "api_key_set": api_key_ok,
                    "database_status": db_status_message
                },
                "message": "Data ingestion service is healthy."
        }), 200
    else:
        return jsonify({
            "status": "error",
            "error": {
                "code": "SERVICE_UNHEALTHY",
                "message": "Data ingestion service is unhealthy or has configuration issues."
            },
            "details": {
                 "api_key_set": api_key_ok, # Still report API key status
                 "database_status": db_status_message
            }
        }), 503

@app.route('/data/fetch', methods=['POST'])
def fetch_data():
    """
    Endpoint to trigger data fetching for specified tickers and date range using Financial Modeling Prep.
    Expects a JSON payload like:
    {
        "tickers": ["AAPL", "MSFT"],
        "start_date": "2020-01-01", 
        "end_date": "2023-01-01"
    }
    Returns adjusted closing prices.
    """
    if not FMP_API_KEY:
        app.logger.error("FMP_API_KEY not configured, cannot fetch data.")
        return jsonify({
            "status": "error", 
            "error": {
                "code": "CONFIGURATION_ERROR",
                "message": "API key for FMP (Financial Modeling Prep) is not configured."
            },
            "details": "Service cannot fetch data without FMP_API_KEY."
        }), 503
    if not db_engine:
        app.logger.error("Database is not available.")
        return jsonify({
            "status": "error",
            "error": {
                "code": "DATABASE_UNAVAILABLE",
                "message": "Database connection is not available."
            },
            "details": "Service cannot store data without a database connection."
        }), 503
    
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({
                "status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Missing JSON payload."}
            }), 400
    except Exception as e: # Handles cases where request.get_json() fails (e.g. not JSON)
        app.logger.error(f"Invalid JSON payload: {e}", exc_info=True)
        return jsonify({
            "status": "error", "error": {"code": "VALIDATION_ERROR", "message": "Invalid JSON payload."}
        }), 400
        
    required_fields = ["tickers", "start_date", "end_date"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": f"Missing required parameters: {', '.join(missing_fields)}."}
        }), 400

    tickers = payload.get('tickers')
    start_date_str = payload.get('start_date')
    end_date_str = payload.get('end_date')

    if not isinstance(tickers, list) or not tickers:
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": "'tickers' must be a non-empty list."}
        }), 400
    if not all(isinstance(t, str) for t in tickers):
        return jsonify({
            "status": "error",
            "error": {"code": "VALIDATION_ERROR", "message": "All items in 'tickers' list must be strings."}
        }), 400

    # Further date validation could be added here (e.g., format, logical range)

    processed_tickers_summary = []
    total_rows_upserted_overall = 0

    for i, ticker_symbol in enumerate(tickers):
        app.logger.info(f"Processing ticker: {ticker_symbol} [{i+1}/{len(tickers)}]")
        ticker_result = {"ticker": ticker_symbol, "status": "pending", "rows_upserted": 0, "details": ""}
        try:
            url = f"{FMP_BASE_URL}/historical-price-full/{ticker_symbol}?from={start_date_str}&to={end_date_str}&apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status()
            fmp_data = response.json()

            if not fmp_data or 'historical' not in fmp_data or not isinstance(fmp_data['historical'], list) or not fmp_data['historical']:
                app.logger.warning(f"No valid historical data from FMP for {ticker_symbol}. Response: {str(fmp_data)[:200]}...")
                ticker_result.update({"status": "no_data_from_api", "details": "No historical data found or data in unexpected format from FMP."})
                processed_tickers_summary.append(ticker_result)
                if i < len(tickers) - 1: time.sleep(1)
                continue
                
            df = pd.DataFrame(fmp_data['historical'])
            if df.empty or 'adjClose' not in df.columns or 'date' not in df.columns:
                app.logger.warning(f"Historical data for {ticker_symbol} (FMP) parsed to empty or malformed DataFrame. Columns: {df.columns}")
                ticker_result.update({"status": "empty_or_malformed_data", "details": "API data parsed to empty or malformed DataFrame."})
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
                app.logger.warning(f"Data for {ticker_symbol} became empty after cleaning. Skipping.")
                ticker_result.update({"status": "empty_after_cleaning", "details": "Data became empty after type conversion or NaN removal."})
                processed_tickers_summary.append(ticker_result)
                if i < len(tickers) - 1: time.sleep(1)
                continue

            upsert_sql = text("""
            INSERT INTO stock_prices (date, ticker, adj_close)
            VALUES (:date, :ticker, :adj_close)
        ON CONFLICT (date, ticker) DO UPDATE SET adj_close = EXCLUDED.adj_close;
            """)
            
            rows_for_upsert = df.to_dict(orient='records')
            with db_engine.connect() as connection:
                connection.execute(upsert_sql, rows_for_upsert) # Execute with list of dicts
                # Note: The execute many with ON CONFLICT might not return individual row counts directly with all DB drivers/SQLAlchemy versions.
                # We are assuming len(rows_for_upsert) is the number of attempted upserts.
            
            current_ticker_rows = len(rows_for_upsert)
            total_rows_upserted_overall += current_ticker_rows
            ticker_result.update({"status": "success", "rows_upserted": current_ticker_rows, "details": f"Successfully processed and initiated upsert for {current_ticker_rows} rows."})
            app.logger.info(f"Successfully initiated upsert for {current_ticker_rows} rows for {ticker_symbol}.")
            processed_tickers_summary.append(ticker_result)

            if i < len(tickers) - 1: time.sleep(1)

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else 'N/A'
            err_code = "API_CLIENT_ERROR" if isinstance(status_code, int) and 400 <= status_code < 500 else "API_SERVER_ERROR"
            log_msg = f"HTTP error for {ticker_symbol} (FMP): {http_err}. Status: {status_code}. Response: {http_err.response.text[:200] if http_err.response else 'No response text'}..."
            app.logger.error(log_msg)
            ticker_result.update({"status": "api_http_error", "details": {"code": err_code, "message": str(http_err), "fmp_status_code": status_code}})
            processed_tickers_summary.append(ticker_result)
        except requests.exceptions.RequestException as req_err:
            app.logger.error(f"Request error (e.g., connection, timeout) for {ticker_symbol} (FMP): {req_err}", exc_info=True)
            ticker_result.update({"status": "api_request_error", "details": {"code": "API_CONNECTION_ERROR", "message": str(req_err)}})
            processed_tickers_summary.append(ticker_result)
        except Exception as e: # Catch-all for other errors during a specific ticker's processing
            app.logger.error(f"General error processing {ticker_symbol}: {str(e)}", exc_info=True)
            ticker_result.update({"status": "processing_error", "details": {"code": "INTERNAL_TICKER_PROCESSING_ERROR", "message": str(e)}})
            processed_tickers_summary.append(ticker_result)
        
    final_message = f"Data fetching process completed. Total rows upserted attempt count: {total_rows_upserted_overall}."
    if total_rows_upserted_overall == 0 and any(t['status'] != 'success' for t in processed_tickers_summary):
        final_message = "Data fetching process completed, but no new data was successfully processed or upserted."
        
    return jsonify({
        "status": "success", 
        "data": {
            "total_rows_upserted_attempted": total_rows_upserted_overall,
            "summary_per_ticker": processed_tickers_summary
        },
        "message": final_message
    }), 200

if __name__ == '__main__':
    # Enable debug logging for the app
    app.run(host='0.0.0.0', port=5001, debug=True) 