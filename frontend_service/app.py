import streamlit as st
import requests
import os
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

# API Gateway base URL from environment variable
# This should match the value set in docker-compose.yml: http://api_gateway_service:5000
API_GATEWAY_BASE_URL = os.getenv('API_GATEWAY_URL', 'http://api_gateway_service:5000')

# Sample list of popular stock tickers for autocomplete suggestions
POPULAR_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", 
    "V", "WMT", "DIS", "NFLX", "PYPL", "INTC", "AMD", "BA", "KO", 
    "MCD", "CSCO", "VZ", "T", "IBM", "GE", "XOM", "CVX", "PFE",
    "JNJ", "MRK", "PG", "UNH", "HD", "GS", "CAT", "MMM", "NKE"
]

# Initialize session state for persisting user inputs
if 'tickers_str' not in st.session_state:
    st.session_state['tickers_str'] = "AAPL,MSFT"
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = ""
if 'sequence_length' not in st.session_state:
    st.session_state['sequence_length'] = 60
if 'hidden_layer_size' not in st.session_state:
    st.session_state['hidden_layer_size'] = 50
if 'num_layers' not in st.session_state:
    st.session_state['num_layers'] = 2
if 'num_epochs' not in st.session_state:
    st.session_state['num_epochs'] = 10
if 'risk_free_rate' not in st.session_state:
    st.session_state['risk_free_rate'] = 0.02

# Function to suggest tickers as user types
def suggest_tickers(input_text):
    """Suggest tickers based on user input."""
    if not input_text:
        return []
    
    # Extract the last partial ticker from comma-separated input
    parts = input_text.split(',')
    current_input = parts[-1].strip().upper()
    
    # No suggestions for empty current input
    if not current_input:
        return []
        
    # Match tickers that start with the current input
    suggestions = [ticker for ticker in POPULAR_TICKERS if ticker.startswith(current_input)]
    
    # If no exact start matches, try to find tickers containing the input
    if not suggestions:
        suggestions = [ticker for ticker in POPULAR_TICKERS if current_input in ticker]
        
    # Format suggestions to show what would be inserted
    formatted_suggestions = []
    if len(parts) > 1:
        base = ','.join(parts[:-1]) + ','
        formatted_suggestions = [base + suggestion for suggestion in suggestions]
    else:
        formatted_suggestions = suggestions
        
    return formatted_suggestions

# Function to fetch available models
def fetch_available_models():
    try:
        models_url = f"{API_GATEWAY_BASE_URL}/api/v1/model/list"
        response = requests.get(models_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and "data" in data and "models" in data["data"]:
                return data["data"]["models"]
        
        return {}  # Return empty dict if any issues
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return {}

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

    # Move ticker input and suggestions outside the form
    tickers_input = st.text_input(
        "Tickers (comma-separated, e.g., AAPL,MSFT,GOOG)", 
        st.session_state['tickers_str'],
        key="ingest_tickers"
    )
    
    # Show ticker suggestions as the user types
    if tickers_input:
        suggestions = suggest_tickers(tickers_input)
        if suggestions:
            st.caption("Suggestions (click to select):")
            cols = st.columns(min(4, len(suggestions)))
            for i, suggestion in enumerate(suggestions[:4]):  # Limit to 4 suggestions
                with cols[i]:
                    if st.button(suggestion, key=f"suggest_{i}"):
                        st.session_state['tickers_str'] = suggestion
                        st.rerun()

    with st.form("data_fetch_form"):
        # Use session state value in the form
        st.write(f"Selected tickers: **{st.session_state['tickers_str']}**")
        start_date = st.date_input("Start Date", default_start_date)
        end_date = st.date_input("End Date", default_end_date)
        
        submitted = st.form_submit_button("Fetch Data")

    if submitted:
        if not st.session_state['tickers_str']:
            st.warning("Please enter at least one ticker.")
        else:
            tickers_list = [ticker.strip().upper() for ticker in st.session_state['tickers_str'].split(',')]
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
                        
                        # Visualize the fetched data if data was returned
                        if response_data.get("new_rows_by_ticker", {}) and any(response_data["new_rows_by_ticker"].values()):
                            st.subheader("Data Visualization")
                            st.info("Fetching latest data for visualization...")
                            
                            # Fetch the most recent data for visualization
                            try:
                                # This would need an endpoint to fetch the actual price data 
                                # For now, just show a placeholder message
                                st.info("Visualization of historical prices will be implemented in future updates.")
                                # Placeholder for visualization code
                            except Exception as viz_error:
                                st.warning(f"Could not visualize data: {viz_error}")
                            
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

    # Fetch existing models for reference
    with st.expander("View Existing Models", expanded=False):
        if st.button("Refresh Model List"):
            existing_models = fetch_available_models()
            if existing_models:
                st.success(f"Found {len(existing_models)} existing models")
                for model_name, model_data in existing_models.items():
                    st.subheader(f"Model: {model_name}")
                    st.write(f"Tickers: {', '.join(model_data.get('tickers', []))}")
                    st.write(f"Created: {model_data.get('creation_date', 'Unknown')}")
                    with st.expander("Configuration Details"):
                        st.json(model_data.get('ticker_details', {}))
            else:
                st.info("No existing models found or could not fetch model list.")
    
    # Move ticker input and suggestions outside the form
    train_tickers_input = st.text_input(
        "Tickers to train (comma-separated)", 
        st.session_state['tickers_str'], 
        key="train_tickers"
    )
    
    # Show ticker suggestions as the user types
    if train_tickers_input:
        suggestions = suggest_tickers(train_tickers_input)
        if suggestions:
            st.caption("Suggestions (click to select):")
            cols = st.columns(min(4, len(suggestions)))
            for i, suggestion in enumerate(suggestions[:4]):  # Limit to 4 suggestions
                with cols[i]:
                    if st.button(suggestion, key=f"train_suggest_{i}"):
                        st.session_state['tickers_str'] = suggestion
                        st.rerun()

    with st.form("model_train_form"):
        st.subheader("Data Selection")
        # Use session state value in the form
        st.write(f"Selected tickers: **{st.session_state['tickers_str']}**")
        
        train_start_date = st.date_input("Training Data Start Date", default_start_date_train, key="train_start_date")
        train_end_date = st.date_input("Training Data End Date", default_end_date_train, key="train_end_date")
        
        st.subheader("Model Configuration")
        # Generate default model name based on current date and tickers
        default_model_name = f"lstm_{datetime.date.today().strftime('%Y%m%d')}"
        if st.session_state['tickers_str']:
            ticker_initials = ''.join([t[0] for t in st.session_state['tickers_str'].split(',')[:3]])
            default_model_name += f"_{ticker_initials}"
            
        model_name = st.text_input("Model Name", 
                                   st.session_state.get('model_name', default_model_name), 
                                   help="Enter the name for the model to be trained.")
        
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.number_input("Sequence Length", 
                                            min_value=10, 
                                            max_value=200, 
                                            value=st.session_state['sequence_length'], 
                                            step=1)
            num_epochs = st.number_input("Number of Epochs", 
                                       min_value=1, 
                                       max_value=1000, 
                                       value=st.session_state['num_epochs'], 
                                       step=1)
        with col2:
            hidden_layer_size = st.number_input("Hidden Layer Size", 
                                               min_value=10, 
                                               max_value=512, 
                                               value=st.session_state['hidden_layer_size'], 
                                               step=1)
            num_layers = st.number_input("Number of LSTM Layers", 
                                       min_value=1, 
                                       max_value=10, 
                                       value=st.session_state['num_layers'], 
                                       step=1)
            
        train_submitted = st.form_submit_button("Train Model(s)")

    if train_submitted:
        if not st.session_state['tickers_str']:
            st.warning("Please enter at least one ticker to train.")
        elif not model_name:
            st.warning("Please enter a model name.")
        else:
            # Update session state
            st.session_state['model_name'] = model_name
            st.session_state['sequence_length'] = sequence_length
            st.session_state['num_epochs'] = num_epochs
            st.session_state['hidden_layer_size'] = hidden_layer_size
            st.session_state['num_layers'] = num_layers
            
            train_tickers_list = [ticker.strip().upper() for ticker in st.session_state['tickers_str'].split(',')]
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
            with st.spinner(f"Training model(s) for {train_tickers_input} with model name {model_name}. This may take a while..."):
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
                            training_results = []
                            for result in response_data["results"]:
                                st.subheader(f"Ticker: {result['ticker']}")
                                st.text(f"Status: {result['status']}")
                                if "message" in result: st.info(result["message"])
                                if "model_path" in result: st.text(f"Model saved to: {result['model_path']}")
                                if "scaler_path" in result: st.text(f"Scaler saved to: {result['scaler_path']}")
                                if "config_path" in result: st.text(f"Config saved to: {result['config_path']}")
                                
                                # Collect metrics for visualization if available
                                if "test_loss" in result:
                                    st.metric(label="Test MSE Loss", value=f"{result['test_loss']:.6f}")
                                    training_results.append({
                                        "ticker": result['ticker'],
                                        "test_loss": result['test_loss']
                                    })
                            
                            # Create a simple bar chart of test loss values if available
                            if training_results:
                                df_results = pd.DataFrame(training_results)
                                if not df_results.empty and 'test_loss' in df_results.columns:
                                    st.subheader("Test Loss by Ticker")
                                    fig = px.bar(df_results, x='ticker', y='test_loss', 
                                                title='Model Test Loss (MSE) by Ticker',
                                                labels={'ticker': 'Ticker', 'test_loss': 'Test MSE Loss'})
                                    st.plotly_chart(fig)
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

    # Fetch available models for the dropdown
    models_data = fetch_available_models()
    available_model_names = list(models_data.keys()) if models_data else []
    
    # Move ticker input and suggestions outside the form
    opt_tickers_input = st.text_input(
        "Tickers for Portfolio (comma-separated)", 
        st.session_state['tickers_str'], 
        key="opt_tickers"
    )
    
    # Show ticker suggestions as the user types
    if opt_tickers_input:
        suggestions = suggest_tickers(opt_tickers_input)
        if suggestions:
            st.caption("Suggestions (click to select):")
            cols = st.columns(min(4, len(suggestions)))
            for i, suggestion in enumerate(suggestions[:4]):  # Limit to 4 suggestions
                with cols[i]:
                    if st.button(suggestion, key=f"opt_suggest_{i}"):
                        st.session_state['tickers_str'] = suggestion
                        st.rerun()

    with st.form("portfolio_optimize_form"):
        st.subheader("Asset & Model Selection")
        # Use session state value in the form
        st.write(f"Selected tickers: **{st.session_state['tickers_str']}**")
        
        # Display model selection dropdown if models are available
        if available_model_names:
            opt_model_name = st.selectbox(
                "Select Model for Predictions", 
                options=[""] + available_model_names,
                index=0 if not st.session_state.get('model_name') in available_model_names else available_model_names.index(st.session_state.get('model_name'))+1,
                key="opt_model_name"
            )
            
            # Show model details if one is selected
            if opt_model_name and opt_model_name in models_data:
                model_info = models_data[opt_model_name]
                with st.expander("Selected Model Details"):
                    st.write(f"**Model Name:** {opt_model_name}")
                    st.write(f"**Tickers Available:** {', '.join(model_info.get('tickers', []))}")
                    st.write(f"**Created:** {model_info.get('creation_date', 'Unknown')}")
                    
                    # Get the configuration of the first ticker as reference
                    if model_info.get('tickers') and model_info.get('ticker_details'):
                        first_ticker = model_info['tickers'][0]
                        ticker_details = model_info['ticker_details'].get(first_ticker, {})
                        if 'config' in ticker_details:
                            st.write("**Model Configuration:**")
                            st.json(ticker_details['config'])
        else:
            # If no models available, show text input
            opt_model_name = st.text_input("Model Name to use for predictions (must be an existing, trained model name)", 
                                          st.session_state.get('model_name', ""), 
                                          key="opt_model_name_text")
            if not available_model_names:
                st.warning("Could not fetch available models. Please enter a model name manually.")
        
        st.subheader("Prediction Parameters")
        opt_sequence_length = st.number_input("Sequence Length for fetching prediction input", 
                                            min_value=10, 
                                            max_value=200, 
                                            value=st.session_state['sequence_length'], 
                                            step=1, 
                                            key="opt_seq_len")
        
        st.subheader("Historical Data Range for Covariance Matrix")
        # Default to a recent period for covariance calculation
        default_cov_end_date = datetime.date.today()
        default_cov_start_date = default_cov_end_date - datetime.timedelta(days=2*365) # Approx 2 years for covariance
        cov_start_date = st.date_input("Start Date (for Covariance)", default_cov_start_date, key="cov_start_date")
        cov_end_date = st.date_input("End Date (for Covariance)", default_cov_end_date, key="cov_end_date")

        st.subheader("Optimization Parameters")
        risk_free_rate = st.number_input("Annual Risk-Free Rate (e.g., 0.02 for 2%)", 
                                       min_value=0.0, 
                                       max_value=0.5, 
                                       value=st.session_state['risk_free_rate'], 
                                       step=0.001, 
                                       format="%.4f")
        
        # Add more optimization objectives as options
        objective_function = st.selectbox(
            "Optimization Objective", 
            ["Maximize Sharpe Ratio"], 
            key="opt_objective"
        )
        
        # Make sure the submit button is inside the form
        optimize_submitted = st.form_submit_button("Optimize Portfolio")

    # This block should be after the form has ended
    if optimize_submitted:
        if not st.session_state['tickers_str']:
            st.warning("Please enter at least one ticker for the portfolio.")
        elif not opt_model_name:
            st.warning("Please enter or select a model name to use for predictions.")
        else:
            # Update session state
            st.session_state['model_name'] = opt_model_name
            st.session_state['sequence_length'] = opt_sequence_length
            st.session_state['risk_free_rate'] = risk_free_rate
            
            opt_tickers_list = [ticker.strip().upper() for ticker in st.session_state['tickers_str'].split(',')]
            
            # Check if we should proceed with optimization
            proceed_with_optimization = True
            
            # Validate that all tickers are available in the selected model
            if opt_model_name in models_data:
                available_tickers = models_data[opt_model_name].get('tickers', [])
                unavailable_tickers = [ticker for ticker in opt_tickers_list if ticker not in available_tickers]
                
                if unavailable_tickers:
                    st.error(f"The following tickers are not available in model '{opt_model_name}': {', '.join(unavailable_tickers)}")
                    st.info(f"Available tickers in this model: {', '.join(available_tickers)}")
                    st.info("Please either select only available tickers or train a new model that includes all required tickers.")
                    proceed_with_optimization = False
            
            # Only proceed if all validations pass
            if proceed_with_optimization:
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
                with st.spinner(f"Optimizing portfolio for {', '.join(opt_tickers_list)} using model {opt_model_name}. This may take a moment..."):
                    st.json(payload) # Show what's being sent
                    try:
                        # Use the API Gateway instead of direct connection
                        optimize_url = f"{API_GATEWAY_BASE_URL}/api/v1/optimize"
                        
                        # Try with multiple retry attempts and increasing timeout
                        max_retries = 3
                        current_retry = 0
                        last_error = None
                        
                        while current_retry < max_retries:
                            try:
                                # Increase timeout with each retry
                                timeout = 60 * (current_retry + 2)  # 120s, 180s, 240s
                                
                                # On retries, use simplified payload
                                if current_retry > 0:
                                    st.info(f"Retry attempt {current_retry} with {timeout}s timeout...")
                                    retry_payload = payload.copy()
                                    retry_payload["simplified_response"] = True
                                    response = requests.post(optimize_url, json=retry_payload, timeout=timeout)
                                else:
                                    response = requests.post(optimize_url, json=payload, timeout=timeout)
                                
                                # If we get here, the request succeeded
                                break
                            except requests.exceptions.RequestException as e:
                                last_error = e
                                st.warning(f"Request attempt {current_retry+1} failed: {e}. Retrying...")
                                current_retry += 1
                        
                        # If all retries failed, raise the last error
                        if current_retry == max_retries:
                            raise last_error
                        
                        response_status_code = response.status_code
                        
                        try:
                            response_data = response.json()
                        except requests.exceptions.JSONDecodeError as e:
                            response_data = None
                            st.error(f"Failed to decode JSON response: {e}")
                            st.text(f"Partial response content: {response.text[:1000] if response.text else 'No content'}")
                            
                            # Try to recover by sending a simplified request
                            st.warning("Attempting recovery with simplified request format...")
                            recovery_payload = payload.copy()
                            recovery_payload["simplified_response"] = True
                            
                            try:
                                recovery_response = requests.post(optimize_url, json=recovery_payload, timeout=60)
                                response_data = recovery_response.json()
                                st.success("Recovery successful - fetched simplified response")
                            except Exception as recovery_err:
                                st.error(f"Recovery attempt failed: {recovery_err}")
                                raise e

                        if response_data:
                            st.subheader("Optimization Results:")
                            if response_status_code == 200:
                                st.success(response_data.get("message", "Optimization completed!"))
                                
                                if "optimized_weights" in response_data:
                                    weights_data = response_data["optimized_weights"]
                                    
                                    # Convert to DataFrame for display
                                    weights_df = pd.DataFrame(list(weights_data.items()), columns=['Ticker', 'Weight'])
                                    weights_df['Weight'] = weights_df['Weight'] * 100  # Convert to percentage
                                    
                                    # Display weights table
                                    st.write("**Optimized Weights:**")
                                    st.dataframe(weights_df.set_index('Ticker'))
                                    
                                    # Create pie chart of portfolio weights
                                    fig = px.pie(weights_df, values='Weight', names='Ticker', 
                                                title='Optimized Portfolio Allocation',
                                                labels={'Weight': 'Weight (%)'})
                                    # Add percentage to hover text
                                    fig.update_traces(textinfo='percent+label')
                                    st.plotly_chart(fig)
                                
                                if "metrics" in response_data:
                                    st.write("**Portfolio Metrics (based on model predictions):**")
                                    metrics = response_data["metrics"]
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(label="Expected Annual Return", 
                                                 value=f"{metrics.get('portfolio_expected_annual_return_from_model', 0)*100:.2f}%")
                                    with col2:
                                        st.metric(label="Expected Annual Volatility", 
                                                 value=f"{metrics.get('portfolio_expected_annual_volatility', 0)*100:.2f}%")
                                    with col3:
                                        st.metric(label="Sharpe Ratio", 
                                                 value=f"{metrics.get('portfolio_sharpe_ratio_from_model', 0):.4f}")
                                    
                                    # Create expected return vs volatility chart if data available
                                    if "portfolio_expected_annual_return_from_model" in metrics and "portfolio_expected_annual_volatility" in metrics:
                                        st.subheader("Portfolio on Risk-Return Spectrum")
                                        
                                        # Create a scatter plot with a single point representing the optimized portfolio
                                        risk_return_df = pd.DataFrame({
                                            'Asset': ['Optimized Portfolio'],
                                            'Volatility (%)': [metrics.get('portfolio_expected_annual_volatility', 0) * 100],
                                            'Return (%)': [metrics.get('portfolio_expected_annual_return_from_model', 0) * 100]
                                        })
                                        
                                        fig = px.scatter(risk_return_df, x='Volatility (%)', y='Return (%)', 
                                                        text='Asset', size=[15], 
                                                        title='Expected Return vs Risk',
                                                        labels={'Volatility (%)': 'Annual Volatility (%)', 
                                                               'Return (%)': 'Expected Annual Return (%)'})
                                        
                                        # Add a line from origin with slope = Sharpe ratio (without risk-free rate)
                                        sharpe = metrics.get('portfolio_sharpe_ratio_from_model', 0)
                                        risk_free = risk_free_rate * 100  # Convert to percentage
                                        
                                        # Add a marker for risk-free rate
                                        fig.add_trace(go.Scatter(
                                            x=[0], 
                                            y=[risk_free],
                                            mode='markers+text',
                                            name='Risk-Free Rate',
                                            text=['Risk-Free Rate'],
                                            marker=dict(size=10, color='green'),
                                            textposition="top right"
                                        ))
                                        
                                        # Add a line connecting risk-free rate to optimized portfolio
                                        fig.add_trace(go.Scatter(
                                            x=[0, risk_return_df['Volatility (%)'].values[0]],
                                            y=[risk_free, risk_return_df['Return (%)'].values[0]],
                                            mode='lines',
                                            name='Capital Allocation Line',
                                            line=dict(color='green', dash='dash')
                                        ))
                                        
                                        # Customize layout
                                        fig.update_layout(
                                            showlegend=True,
                                            xaxis=dict(rangemode='tozero'),
                                            yaxis=dict(rangemode='tozero')
                                        )
                                        
                                        st.plotly_chart(fig)
                                
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
                        st.error(f"Error connecting to Portfolio Optimization Service: {e}")
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