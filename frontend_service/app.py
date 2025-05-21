import streamlit as st
import requests
import os
import datetime
import pandas as pd

# API Gateway base URL from environment variable
# This should match the value set in docker-compose.yml: http://api_gateway_service:5000
API_GATEWAY_BASE_URL = os.getenv('API_GATEWAY_URL', 'http://api_gateway_service:5000')

st.set_page_config(layout="wide")
st.title('📈 Portfolio Optimization System')

st.sidebar.header('Navigation')
page = st.sidebar.radio("Choose a page", ["Home", "Data Ingestion", "Model Training", "Portfolio Optimization"])

if page == "Home":
    st.header('Welcome!')
    st.write("Use the navigation on the left to access different services of the Portfolio Optimization System.")
    st.markdown("""
        **Services:**
        - **Data Ingestion**: Fetch historical stock data from Financial Modeling Prep API and store it.
        - **Model Training**: Train LSTM models for stock price prediction using the stored data.
        - **Portfolio Optimization**: Optimize portfolio weights based on model predictions or historical data.
        
        All services are accessible via an API Gateway.
    """)

    if st.button('Check API Gateway Health'):
        try:
            # The API Gateway's own health endpoint is at its root
            health_check_url = f"{API_GATEWAY_BASE_URL}/health" 
            response = requests.get(health_check_url, timeout=5)
            response.raise_for_status() 
            health_status = response.json()
            st.success(f"API Gateway is healthy: {health_status.get('message', 'OK')}")
            st.json(health_status)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API Gateway: {e}")
            st.info(f"Attempted to reach: {health_check_url}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

elif page == "Data Ingestion":
    st.header("Data Ingestion Service")
    st.write("Fetch historical stock data and store it in the database via the API Gateway.")

    # Default to a recent period
    default_end_date = datetime.date.today()
    default_start_date = default_end_date - datetime.timedelta(days=3*365) # Approx 3 years back

    with st.form("data_fetch_form"):
        tickers_str = st.text_input("Tickers (comma-separated, e.g., AAPL,MSFT,GOOG)", "AAPL,MSFT")
        start_date = st.date_input("Start Date", default_start_date)
        end_date = st.date_input("End Date", default_end_date)
        
        submitted = st.form_submit_button("Fetch Data")

    if submitted:
        if not tickers_str:
            st.warning("Please enter at least one ticker.")
        else:
            tickers_list = [ticker.strip().upper() for ticker in tickers_str.split(',')]
            payload = {
                "tickers": tickers_list,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
            
            st.write("Sending request to API Gateway...")
            st.json(payload) # Show what's being sent

            try:
                fetch_url = f"{API_GATEWAY_BASE_URL}/api/v1/data/fetch"
                response = requests.post(fetch_url, json=payload, timeout=300) # Increased timeout for potentially long fetches
                
                if response.status_code == 200 or response.status_code == 201:
                    response_data = response.json()
                    st.success(response_data.get("message", "Data fetched successfully!"))
                    if "new_rows_by_ticker" in response_data:
                        st.write("New rows added:")
                        st.json(response_data["new_rows_by_ticker"])
                    elif "detail" in response_data: # Handle cases where data_ingestion returns details without new_rows
                         st.info(response_data["detail"])
                else:
                    try:
                        error_data = response.json()
                        st.error(f"Error from service (HTTP {response.status_code}): {error_data.get('error', response.text)}")
                        if "details" in error_data: st.json(error_data["details"])
                    except requests.exceptions.JSONDecodeError:
                        st.error(f"Error fetching data (HTTP {response.status_code}): {response.text}")
                
                st.subheader("Full Response:")
                try:
                    st.json(response.json())
                except requests.exceptions.JSONDecodeError:
                    st.text(response.text)

            except requests.exceptions.Timeout:
                st.error(f"Request timed out after 300 seconds when trying to reach {fetch_url}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API Gateway at {fetch_url}: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

elif page == "Model Training":
    st.header("Model Training Service")
    st.write("Train time-series prediction models for selected tickers using data from the database.")
    st.info("ℹ️ Model training can be a time-consuming process, especially with many epochs or large datasets. Please be patient after submitting.")

    default_end_date_train = datetime.date.today()
    default_start_date_train = default_end_date_train - datetime.timedelta(days=3*365) # Approx 3 years back

    with st.form("model_train_form"):
        st.subheader("Data Selection")
        train_tickers_str = st.text_input("Tickers to train (comma-separated)", "AAPL,MSFT", key="train_tickers")
        train_start_date = st.date_input("Training Data Start Date", default_start_date_train, key="train_start_date")
        train_end_date = st.date_input("Training Data End Date", default_end_date_train, key="train_end_date")
        
        st.subheader("Model Configuration")
        # User must explicitly enter the model name for training.
        model_name = st.text_input("Model Name (e.g., lstm_v3_frontend or YOUR_CUSTOM_NAME)", "", help="Enter the name for the model to be trained.")
        
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.number_input("Sequence Length", min_value=10, max_value=200, value=60, step=1)
            num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=10, step=1)
        with col2:
            hidden_layer_size = st.number_input("Hidden Layer Size", min_value=10, max_value=512, value=50, step=1)
            num_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=10, value=2, step=1)
            
        train_submitted = st.form_submit_button("Train Model(s)")

    if train_submitted:
        if not train_tickers_str:
            st.warning("Please enter at least one ticker to train.")
        elif not model_name:
            st.warning("Please enter a model name.")
        else:
            train_tickers_list = [ticker.strip().upper() for ticker in train_tickers_str.split(',')]
            hyperparameters = {
                "sequence_length": sequence_length,
                "num_epochs": num_epochs,
                "hidden_layer_size": hidden_layer_size,
                "num_layers": num_layers,
                "input_size": 1,  # Assuming univariate (e.g., Adj Close)
                "output_size": 1
            }
            payload = {
                "tickers": train_tickers_list,
                "start_date": train_start_date.strftime("%Y-%m-%d"),
                "end_date": train_end_date.strftime("%Y-%m-%d"),
                "model_name": model_name,
                "hyperparameters": hyperparameters
            }
            
            st.write("Sending training request to API Gateway...")
            with st.spinner(f"Training model(s) for {train_tickers_str} with model name {model_name}. This may take a while..."):
                st.json(payload) # Show what's being sent
                try:
                    train_url = f"{API_GATEWAY_BASE_URL}/api/v1/model/train"
                    # Model training can take very long, use a very long timeout
                    # The API gateway itself has a 900s timeout for this route.
                    # Streamlit request should be slightly less to allow gateway to respond if it times out.
                    response = requests.post(train_url, json=payload, timeout=600) 
                    
                    if response.status_code == 200 or response.status_code == 201:
                        response_data = response.json()
                        st.success(response_data.get("message", "Model training submitted successfully!"))
                        if "results" in response_data:
                            st.write("Training Results:")
                            for result in response_data["results"]:
                                st.subheader(f"Ticker: {result['ticker']}")
                                st.text(f"Status: {result['status']}")
                                if "message" in result: st.info(result["message"])
                                if "model_path" in result: st.text(f"Model saved to: {result['model_path']}")
                                if "scaler_path" in result: st.text(f"Scaler saved to: {result['scaler_path']}")
                                if "config_path" in result: st.text(f"Config saved to: {result['config_path']}")
                                if "test_loss" in result: st.metric(label="Test MSE Loss", value=f"{result['test_loss']:.6f}")
                    else:
                        try:
                            error_data = response.json()
                            st.error(f"Error from training service (HTTP {response.status_code}): {error_data.get('error', {}).get('message', response.text)}")
                            if "original_exception" in error_data.get('error', {}): st.caption(f"Details: {error_data['error']['original_exception']}")
                        except requests.exceptions.JSONDecodeError:
                            st.error(f"Error training model (HTTP {response.status_code}): {response.text}")
                    
                    st.subheader("Full Response from Model Training Service:")
                    try:
                        st.json(response.json())
                    except requests.exceptions.JSONDecodeError:
                        st.text(response.text)

                except requests.exceptions.Timeout:
                    st.error(f"Request timed out after 600 seconds when trying to reach {train_url}. The training job might still be running on the server if it started.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API Gateway at {train_url}: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during model training request: {e}")

elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization Service")
    st.write("Optimize your portfolio weights based on model predictions or historical data.")
    st.info("ℹ️ This service will fetch predictions from the Model Training service and then run optimization.")

    # Get available models to choose from by listing directories in the model_training_service volume
    # This is a simplified approach. A dedicated endpoint in model_training_service to list models would be more robust.
    # For now, we rely on the user knowing the model_name or we can prefill with the last trained one if we store it.
    # Let's assume user enters model name for now.

    with st.form("portfolio_optimize_form"):
        st.subheader("Asset & Model Selection")
        opt_tickers_str = st.text_input("Tickers for Portfolio (comma-separated)", "AAPL,MSFT", key="opt_tickers")
        # User must explicitly enter the trained model name.
        opt_model_name = st.text_input("Model Name to use for predictions (must be an existing, trained model name)", "", key="opt_model_name")
        
        st.subheader("Prediction Parameters")
        opt_sequence_length = st.number_input("Sequence Length for fetching prediction input", min_value=10, max_value=200, value=60, step=1, key="opt_seq_len")
        
        st.subheader("Historical Data Range for Covariance Matrix")
        # Default to a recent period for covariance calculation
        default_cov_end_date = datetime.date.today()
        default_cov_start_date = default_cov_end_date - datetime.timedelta(days=2*365) # Approx 2 years for covariance
        cov_start_date = st.date_input("Start Date (for Covariance)", default_cov_start_date, key="cov_start_date")
        cov_end_date = st.date_input("End Date (for Covariance)", default_cov_end_date, key="cov_end_date")

        st.subheader("Optimization Parameters")
        risk_free_rate = st.number_input("Annual Risk-Free Rate (e.g., 0.02 for 2%)", min_value=0.0, max_value=0.5, value=0.02, step=0.001, format="%.4f")
        # Currently, optimization service only supports Sharpe ratio. Could add more options here later.
        objective_function = st.selectbox("Optimization Objective", ["Maximize Sharpe Ratio"], key="opt_objective") 

        optimize_submitted = st.form_submit_button("Optimize Portfolio")

    if optimize_submitted:
        if not opt_tickers_str:
            st.warning("Please enter at least one ticker for the portfolio.")
        elif not opt_model_name:
            st.warning("Please enter the model name to use for predictions.")
        else:
            opt_tickers_list = [ticker.strip().upper() for ticker in opt_tickers_str.split(',')]
            
            prediction_params = {
                "sequence_length": opt_sequence_length
            }
            
            payload = {
                "tickers": opt_tickers_list,
                "model_name": opt_model_name,
                "start_date": cov_start_date.strftime("%Y-%m-%d"),
                "end_date": cov_end_date.strftime("%Y-%m-%d"),
                "prediction_parameters": prediction_params,
                "optimization_parameters": {
                    "risk_free_rate": risk_free_rate,
                    "objective": objective_function # e.g., "Maximize Sharpe Ratio"
                }
            }
            
            st.write("Sending optimization request to API Gateway...")
            with st.spinner(f"Optimizing portfolio for {opt_tickers_str} using model {opt_model_name}. This may take a moment..."):
                st.json(payload) # Show what's being sent
                try:
                    # Bypass the API Gateway and connect directly to the portfolio optimization service
                    optimize_url_direct = "http://portfolio_optimization_service:5000/optimize"
                    
                    st.info("Sending direct request to the Portfolio Optimization Service...")
                    
                    # Try first with regular timeout
                    try:
                        response = requests.post(optimize_url_direct, json=payload, timeout=120)
                    except requests.exceptions.RequestException as e:
                        st.warning(f"First request failed: {e}. Trying simplified request...")
                        
                        # Add a simplified flag to get minimal response
                        payload["simplified_response"] = True
                        response = requests.post(optimize_url_direct, json=payload, timeout=60)
                    
                    response_status_code = response.status_code
                    
                    try:
                        response_data = response.json()
                    except requests.exceptions.JSONDecodeError as e:
                        response_data = None
                        st.error(f"Failed to decode JSON response: {e}")
                        st.text(f"Partial response content: {response.text[:1000] if response.text else 'No content'}")

                    if response_data:
                        st.subheader("Optimization Results:")
                        if response_status_code == 200:
                            st.success(response_data.get("message", "Optimization completed!"))
                            
                            if "optimized_weights" in response_data:
                                st.write("**Optimized Weights:**")
                                st.dataframe(pd.DataFrame(list(response_data["optimized_weights"].items()), columns=['Ticker', 'Weight']).set_index('Ticker'))
                            
                            if "metrics" in response_data:
                                st.write("**Portfolio Metrics (based on model predictions):**")
                                metrics = response_data["metrics"]
                                st.metric(label="Expected Annual Return (from model)", value=f"{metrics.get('portfolio_expected_annual_return_from_model', 0)*100:.2f}%")
                                st.metric(label="Expected Annual Volatility", value=f"{metrics.get('portfolio_expected_annual_volatility', 0)*100:.2f}%")
                                st.metric(label="Sharpe Ratio (from model)", value=f"{metrics.get('portfolio_sharpe_ratio_from_model', 0):.4f}")
                            
                            # Display additional details if available
                            details = response_data.get("details", {})
                            if details:
                                with st.expander("Additional Details"):
                                    if "current_prices" in details:
                                        st.write("**Current Prices Used:**")
                                        st.json(details["current_prices"])
                                    if "predicted_prices" in details:
                                        st.write("**Predicted Next Day Prices:**")
                                        st.json(details["predicted_prices"])
                                    if "predicted_daily_returns" in details:
                                        st.write("**Model Predicted Daily Returns:**")
                                        st.json(details["predicted_daily_returns"])
                                
                        else:
                            st.error(f"Error from optimization service (HTTP {response_status_code}): {response_data.get('error', 'Unknown error')}")
                            if "details" in response_data: st.json(response_data["details"])
                            if "message" in response_data and response_status_code != 200 : st.warning(response_data["message"])

                        st.subheader("Full Response from Optimization Service:")
                        st.json(response_data)

                except requests.exceptions.Timeout:
                    st.error(f"Request timed out after 120 seconds when trying to reach the portfolio optimization service.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to Portfolio Optimization Service at {optimize_url_direct}: {e}")
                    st.info("Try refreshing the page or check if the Portfolio Optimization Service is running properly.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during portfolio optimization request: {e}")
                    st.info("Contact your system administrator for assistance.")

# Placeholder for future functionality
# st.subheader('Fetch Data')
# ... UI elements for data fetching ...

# st.subheader('Train Model')
# ... UI elements for model training ...

# st.subheader('Optimize Portfolio')
# ... UI elements for portfolio optimization ... 