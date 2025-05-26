import streamlit as st
import requests
import os
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import re
import time

# Set a default template for better looking plots
pio.templates.default = "plotly_white"

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
    st.session_state['num_epochs'] = 50
if 'risk_free_rate' not in st.session_state:
    st.session_state['risk_free_rate'] = 0.02
if 'last_training_result' not in st.session_state:
    st.session_state['last_training_result'] = None
if 'training_in_progress' not in st.session_state:
    st.session_state['training_in_progress'] = False

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
    """Fetch available models from the Model Training Service through API Gateway."""
    try:
        # First try through API Gateway
        model_list_url = f"{API_GATEWAY_BASE_URL}/api/v1/model/list"
        response = requests.get(model_list_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Process the response to make it easier to use
            models = {}
            
            # Ensure data is not None and has the expected structure
            if data and isinstance(data, dict) and data.get("status") == "success":
                model_list = data.get("data")
                if model_list and isinstance(model_list, list):
                    for model in model_list:
                        if not isinstance(model, dict):
                            continue
                            
                        model_name = model.get("model_name")
                        if not model_name:
                            continue
                            
                        config_params = model.get("config_params", {}) or {}  # Handle None case
                        model_type = config_params.get("model_type", "lstm").upper()  # Default to LSTM if not specified
                        
                        # Create an entry with model details
                        models[model_name] = {
                            "name": model_name,
                            "display_name": f"{model_name} ({model_type})",
                            "model_type": model_type,
                            "tickers": model.get("tickers_trained_for", []) or [],  # Handle None case
                            "tickers_count": len(model.get("tickers_trained_for", []) or []),
                            "creation_date": model.get("creation_date")
                        }
                    return models
        
        # If API Gateway fails, try direct connection to model training service
        direct_url = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://model_training_service:8000/model/list')
        st.info(f"API Gateway model list failed, trying direct connection to {direct_url}")
        direct_response = requests.get(direct_url, timeout=10)
        
        if direct_response.status_code == 200:
            data = direct_response.json()
            models = {}
            
            if data and isinstance(data, dict) and data.get("status") == "success":
                model_list = data.get("data")
                if model_list and isinstance(model_list, list):
                    for model in model_list:
                        if not isinstance(model, dict):
                            continue
                            
                        model_name = model.get("model_name")
                        if not model_name:
                            continue
                            
                        config_params = model.get("config_params", {}) or {}
                        model_type = config_params.get("model_type", "lstm").upper()
                        
                        models[model_name] = {
                            "name": model_name,
                            "display_name": f"{model_name} ({model_type})",
                            "model_type": model_type,
                            "tickers": model.get("tickers_trained_for", []) or [],
                            "tickers_count": len(model.get("tickers_trained_for", []) or []),
                            "creation_date": model.get("creation_date")
                        }
                    return models
        
        return {}
    except Exception as e:
        st.error(f"Error fetching available models: {e}")
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
    
    # Display last training result if available
    if st.session_state.get('last_training_result'):
        result = st.session_state['last_training_result']
        completion_time = datetime.datetime.fromtimestamp(result['completion_time'])
        
        with st.expander("📊 Last Training Result", expanded=True):
            if result['status'] == 'completed':
                st.success(f"✅ Model '{result['model_name']}' training completed successfully!")
                st.write(f"**Tickers:** {', '.join(result['tickers'])}")
                st.write(f"**Completed at:** {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show training results if available
                if 'response_data' in result and 'data' in result['response_data']:
                    data = result['response_data']['data']
                    
                    # Show overall message
                    if 'overall_message' in data:
                        st.write(f"**Summary:** {data['overall_message']}")
                    
                    # Show per-ticker results
                    if 'training_summary_per_ticker' in data:
                        ticker_summary = data['training_summary_per_ticker']
                        st.write("**Results per Ticker:**")
                        for ticker, summary in ticker_summary.items():
                            status = summary.get('status', 'unknown')
                            message = summary.get('message', 'No message')
                            if status == 'success':
                                st.write(f"  • {ticker}: ✅ {message}")
                            else:
                                st.write(f"  • {ticker}: ❌ {message}")
                
                if st.button("🗑️ Clear Result", key="clear_training_result"):
                    st.session_state['last_training_result'] = None
                    st.rerun()
                    
            elif result['status'] == 'timeout':
                st.warning(f"⏱️ Training for model '{result['model_name']}' timed out")
                st.write(f"**Tickers:** {', '.join(result['tickers'])}")
                st.write(f"**Last checked at:** {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.info("The model may still be training in the background. You can check its status manually.")
                
                if st.button("🔍 Try Training Again", key="retry_training"):
                    st.info("Please submit a new training request above.")
                
                if st.button("🗑️ Clear Result", key="clear_timeout_result"):
                    st.session_state['last_training_result'] = None
                    st.rerun()

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
                    
                    # Replace the inner expander with a collapsible container
                    if st.button(f"Show Config Details for {model_name}", key=f"config_{model_name}"):
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
        
        # Model type selection
        model_type_options = ["lstm", "gru", "transformer", "wavenet"]
        model_type = st.selectbox(
            "Model Architecture",
            options=model_type_options,
            index=model_type_options.index(st.session_state.get('model_type', 'lstm')),
            format_func=lambda x: x.upper(),  # Display in uppercase
            help="Select the neural network architecture to use for time series prediction."
        )
        
        # Update the default model name based on selected model type
        if model_name.startswith(("lstm_", "gru_", "transformer_", "wavenet_")):
            # Replace the prefix with the selected model type
            model_name = f"{model_type}_{model_name.split('_', 1)[1]}"
        
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.number_input("Sequence Length", 
                                            min_value=10, 
                                            max_value=200, 
                                            value=st.session_state.get('sequence_length', 60), 
                                            step=1)
            num_epochs = st.number_input("Number of Epochs", 
                                       min_value=1, 
                                       max_value=1000, 
                                       value=st.session_state.get('num_epochs', 50), 
                                       step=1)
        with col2:
            hidden_layer_size = st.number_input("Hidden Layer Size", 
                                               min_value=10, 
                                               max_value=512, 
                                               value=st.session_state.get('hidden_layer_size', 100), 
                                               step=1)
            num_layers = st.number_input("Number of Layers", 
                                       min_value=1, 
                                       max_value=10, 
                                       value=st.session_state.get('num_layers', 2), 
                                       step=1,
                                       help=f"Number of stacked {model_type.upper()} layers")
        
        # Additional parameters for specific model types
        if model_type == "transformer":
            nhead = st.number_input("Number of Attention Heads", 
                                   min_value=1,
                                   max_value=16,
                                   value=st.session_state.get('nhead', 4),
                                   step=1,
                                   help="Number of attention heads in the transformer model")
            
        train_submitted = st.form_submit_button("Train Model(s)")

    if train_submitted:
        if not st.session_state['tickers_str']:
            st.warning("Please enter at least one ticker to train.")
        elif not model_name:
            st.warning("Please enter a model name.")
        else:
            # Update session state
            st.session_state['model_name'] = model_name
            st.session_state['model_type'] = model_type
            st.session_state['sequence_length'] = sequence_length
            st.session_state['num_epochs'] = num_epochs
            st.session_state['hidden_layer_size'] = hidden_layer_size
            st.session_state['num_layers'] = num_layers
            if model_type == "transformer":
                st.session_state['nhead'] = nhead
            
            train_tickers_list = [ticker.strip().upper() for ticker in st.session_state['tickers_str'].split(',')]
            
            # Create payload with proper structure to match model_training_service expectations
            payload = {
                "tickers": train_tickers_list,
                "start_date": train_start_date.strftime("%Y-%m-%d"),
                "end_date": train_end_date.strftime("%Y-%m-%d"),
                "model_name": model_name,
                "model_type": model_type,
                "sequence_length": sequence_length,
                "epochs": num_epochs,  # Add epochs directly to match TrainRequest model
                "hidden_layer_size": hidden_layer_size,
                "num_layers": num_layers,
                "learning_rate": 0.001,  # Default from model_training_service
                "batch_size": 32,       # Default from model_training_service
                "test_split_size": 0.2  # Default from model_training_service
            }
            
            # Add model-specific parameters
            if model_type == "transformer":
                payload["nhead"] = nhead
            
            st.write("Sending training request to API Gateway...")
            
            # Mark training as in progress and clear any previous results
            st.session_state['training_in_progress'] = True
            st.session_state['last_training_result'] = None
            
            with st.spinner(f"Training {model_type.upper()} model(s) for {train_tickers_input} with model name {model_name}. This may take a while..."):
                st.json(payload) # Show what's being sent
                try:
                    # Use the direct training endpoint for reliable results
                    train_url = f"{API_GATEWAY_BASE_URL}/api/v1/model/train"
                    response = requests.post(train_url, json=payload, timeout=1800)  # 30 minutes timeout for training
                    
                    if response.status_code == 200 or response.status_code == 201:
                        response_data = response.json()
                        st.success("✅ Model training completed successfully!")
                        
                        # Store the training results in session state
                        st.session_state['last_training_result'] = {
                            'model_name': model_name,
                            'tickers': train_tickers_list,
                            'completion_time': time.time(),
                            'response_data': response_data,
                            'status': 'completed'
                        }
                        st.session_state['training_in_progress'] = False
                        
                        # Display training results
                        if "data" in response_data:
                            data = response_data["data"]
                            
                            # Show overall status
                            overall_status = data.get("status_overall", "unknown")
                            overall_message = data.get("overall_message", "Training completed")
                            st.info(f"**Status:** {overall_status}")
                            st.info(f"**Summary:** {overall_message}")
                            
                            # Show per-ticker results
                            if "training_summary_per_ticker" in data:
                                st.subheader("📊 Training Results per Ticker")
                                ticker_summary = data["training_summary_per_ticker"]
                                
                                for ticker, summary in ticker_summary.items():
                                    with st.expander(f"📈 {ticker} Results", expanded=True):
                                        status = summary.get("status", "unknown")
                                        message = summary.get("message", "No message")
                                        
                                        if status == "success":
                                            st.success(f"✅ {message}")
                                        else:
                                            st.error(f"❌ {message}")
                                        
                                        # Show training metrics if available
                                        if "final_train_loss" in summary:
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Final Train Loss", f"{summary['final_train_loss']:.4f}")
                                            with col2:
                                                st.metric("Final Test Loss", f"{summary['final_test_loss']:.4f}")
                                            with col3:
                                                st.metric("Training Time", f"{summary.get('train_time_seconds', 0):.1f}s")
                                        
                                        # Show dataset info if available
                                        if "dataset_size_total" in summary:
                                            st.write(f"**Dataset:** {summary['dataset_size_total']} total samples "
                                                   f"({summary.get('training_set_size', 0)} train, {summary.get('test_set_size', 0)} test)")
                        
                        # Show a button to refresh the model list
                        if st.button("🔄 Refresh Model List", key="refresh_after_training"):
                            st.rerun()
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
                    st.error(f"Request timed out after 60 seconds when trying to reach {train_url}. The training job might still be running on the server if it started.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API Gateway at {train_url}: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during model training request: {e}")

elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization Service")
    st.write("Optimize your portfolio weights based on model predictions or historical data.")
    st.info("ℹ️ This service will fetch predictions from the Model Training service and then run optimization.")

    # Fetch available models for selection
    with st.expander("Available Models", expanded=True):
        models = fetch_available_models()
        if models:
            st.success(f"Found {len(models)} available trained models")
            model_options = []
            model_map = {}  # Map display names to actual model names
            for model_name, model_data in models.items():
                display_name = model_data.get("display_name", model_name)
                model_map[display_name] = model_name
                tickers_str = ", ".join(model_data.get("tickers", []))
                model_options.append(display_name)
                
                # Add creation date if available
                creation_date = model_data.get("creation_date", "")
                creation_info = f" (Created: {creation_date})" if creation_date else ""
                
                st.write(f"**{display_name}**{creation_info}: {tickers_str}")
                
            if st.session_state.get('model_name') not in models:
                # If the current session model isn't in the available models, reset it
                st.session_state['model_name'] = ""
        else:
            st.info("Model list could not be fetched. You can still enter a model name manually below.")
            st.caption("Tip: If you've already trained a model, you can enter its name directly in the field below.")
            model_options = []
            model_map = {}
            
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

    with st.form("portfolio_optimization_form"):
        st.subheader("Portfolio Specification")
        
        # Use session state value in the form
        st.write(f"Selected tickers: **{st.session_state['tickers_str']}**")
        
        # Use model dropdown if we have models
        if model_options:
            model_display_name = st.selectbox(
                "Select a Model for Predictions",
                options=model_options,
                index=model_options.index(st.session_state.get('model_display_name', model_options[0])) if st.session_state.get('model_display_name') in model_options else 0,
                help="Choose a trained model to use for price predictions."
            )
            # Map display name back to actual model name
            opt_model_name = model_map.get(model_display_name, "")
            
            if models and opt_model_name in models:
                # Show model type
                model_type = models[opt_model_name].get("model_type", "LSTM")
                st.info(f"Selected model architecture: {model_type}")
                
                # Show tickers available in this model
                available_tickers = models[opt_model_name].get("tickers", [])
                if available_tickers:
                    st.caption(f"Model includes these tickers: {', '.join(available_tickers)}")
                # Also show the default tickers in the input field if we have a selection
                if not st.session_state['tickers_str']:
                    st.session_state['tickers_str'] = ",".join(available_tickers[:3])  # Just show first 3 for simplicity
        else:
            opt_model_name = st.text_input(
                "Model Name (No models available, enter manually)", 
                st.session_state.get('model_name', ''),
                help="Enter the name of a trained model to use for predictions."
            )
        
        st.subheader("Prediction Parameters")
        opt_sequence_length = st.number_input("Sequence Length for fetching prediction input", 
                                             min_value=10, 
                                             max_value=200, 
                                             value=st.session_state.get('sequence_length', 60), 
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
                                       value=st.session_state.get('risk_free_rate', 0.02), 
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
            if model_options:
                st.session_state['model_display_name'] = model_display_name
            st.session_state['sequence_length'] = opt_sequence_length
            st.session_state['risk_free_rate'] = risk_free_rate
            
            opt_tickers_list = [ticker.strip().upper() for ticker in st.session_state['tickers_str'].split(',')]
            
            # Check if we should proceed with optimization
            proceed_with_optimization = True
            
            # Validate that all tickers are available in the selected model
            if models and opt_model_name in models:
                available_tickers = models[opt_model_name].get("tickers", [])
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
                        
                        # Display results based on the correct response structure
                        response_data = response.json()
                        st.success(response_data.get("message", "Portfolio optimized successfully."))
                        
                        # Debug logging for response data
                        st.write("DEBUG - Response Data Structure:")
                        st.write(response_data)
                        
                        if response.status_code == 200 and "data" in response_data:
                            data = response_data["data"]
                            
                            # Debug logging for weights data
                            st.write("DEBUG - Optimized Weights Data:")
                            st.write(data.get("optimized_weights", "No weights data found"))
                            
                            # Display optimized weights
                            if "optimized_weights" in data:
                                st.subheader("Optimized Portfolio Weights")
                                weights = data["optimized_weights"]
                                
                                # Debug the weights data type and content
                                st.write(f"DEBUG - Weights type: {type(weights)}")
                                st.write(f"DEBUG - Weights content: {weights}")
                                st.write(f"DEBUG - Weights values: {list(weights.values())}")
                                st.write(f"DEBUG - Weights keys: {list(weights.keys())}")
                                
                                # Create a pie chart of the weights
                                try:
                                    # Create a DataFrame for the pie chart with clean values
                                    pie_data = pd.DataFrame({
                                        'Ticker': list(weights.keys()),
                                        'Weight': [float(w) for w in list(weights.values())]
                                    })
                                    
                                    # Format the data - ensure we have non-zero values for display
                                    # Set a minimum threshold for display (e.g., 0.001%)
                                    MIN_DISPLAY_THRESHOLD = 0.00001
                                    pie_data['Weight_for_display'] = pie_data['Weight'].apply(
                                        lambda x: max(x, MIN_DISPLAY_THRESHOLD) if x > 0 else MIN_DISPLAY_THRESHOLD
                                    )
                                    
                                    # Create a direct Plotly figure - most reliable approach
                                    fig = go.Figure()
                                    
                                    # Add the pie trace with explicit configuration
                                    fig.add_trace(go.Pie(
                                        labels=pie_data['Ticker'],
                                        values=pie_data['Weight_for_display'],
                                        textinfo='label+percent',
                                        hoverinfo='label+percent+value',
                                        textposition='inside',
                                        insidetextorientation='radial',
                                        marker=dict(
                                            colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
                                        ),
                                        pull=[0.1 if ticker == 'AAPL' else 0 for ticker in pie_data['Ticker']]
                                    ))
                                    
                                    # Update layout with explicit styling
                                    fig.update_layout(
                                        title={
                                            'text': 'Portfolio Allocation',
                                            'y': 0.95,
                                            'x': 0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top',
                                            'font': {'size': 20}
                                        },
                                        legend={
                                            'orientation': 'h',
                                            'yanchor': 'bottom',
                                            'y': -0.2,
                                            'xanchor': 'center',
                                            'x': 0.5
                                        },
                                        width=600,
                                        height=500,
                                        margin=dict(l=40, r=40, t=80, b=40)
                                    )
                                    
                                    # Use columns to center the chart
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        # Render the chart - force use_container_width=False for more reliable rendering
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Also display as a table
                                    weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
                                    weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
                                    st.table(weights_df)
                                    
                                except Exception as chart_error:
                                    st.error(f"Error creating portfolio visualization: {chart_error}")
                                    
                                    # Show a bar chart as fallback
                                    st.subheader("Portfolio Allocation (Bar Chart)")
                                    weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
                                    weights_df = weights_df.sort_values("Weight", ascending=False)
                                    
                                    # Use st.bar_chart for simplicity
                                    st.bar_chart(
                                        weights_df.set_index("Ticker")["Weight"],
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    # Always show the table
                                    weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
                                    st.table(weights_df)
                            
                            # Display metrics
                            if "metrics" in data:
                                metrics = data["metrics"]
                                st.subheader("Portfolio Metrics")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if metrics.get("portfolio_expected_annual_return_from_model") is not None:
                                        st.metric(
                                            "Expected Annual Return", 
                                            f"{metrics['portfolio_expected_annual_return_from_model']:.2%}"
                                        )
                                
                                with col2:
                                    if metrics.get("portfolio_expected_annual_volatility") is not None:
                                        st.metric(
                                            "Expected Annual Volatility", 
                                            f"{metrics['portfolio_expected_annual_volatility']:.2%}"
                                        )
                                
                                with col3:
                                    if metrics.get("portfolio_sharpe_ratio_from_model") is not None:
                                        st.metric(
                                            "Sharpe Ratio", 
                                            f"{metrics['portfolio_sharpe_ratio_from_model']:.2f}"
                                        )
                            
                            # Display additional details if available
                            if "details" in data and data["details"] is not None:
                                with st.expander("Additional Details", expanded=False):
                                    details = data["details"]
                                    
                                    if "predicted_prices" in details:
                                        st.subheader("Predicted Next-Day Prices")
                                        pred_prices_df = pd.DataFrame(list(details["predicted_prices"].items()), 
                                                                      columns=["Ticker", "Predicted Price"])
                                        st.table(pred_prices_df)
                                    
                                    if "predicted_daily_returns" in details:
                                        st.subheader("Predicted Daily Returns")
                                        pred_returns_df = pd.DataFrame(list(details["predicted_daily_returns"].items()), 
                                                                       columns=["Ticker", "Predicted Return"])
                                        pred_returns_df["Predicted Return"] = pred_returns_df["Predicted Return"].apply(
                                            lambda x: f"{x:.2%}"
                                        )
                                        st.table(pred_returns_df)
                        else:
                            st.warning("No portfolio optimization results found in the response.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to API Gateway at {optimize_url}: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during portfolio optimization: {e}")

# Placeholder for future functionality
# st.subheader('Fetch Data')
# ... UI elements for data fetching ...

# st.subheader('Train Model')
# ... UI elements for model training ...

# st.subheader('Optimize Portfolio')
# ... UI elements for portfolio optimization ... 