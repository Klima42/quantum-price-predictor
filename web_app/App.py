import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add the parent directory to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from quantum_predictor.predictor import MarketPredictor

st.set_page_config(page_title="Quantum Market Predictor", layout="wide")

st.title("Quantum-Inspired Market Predictor")

# File upload
uploaded_file = st.file_uploader("Upload market data (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        lookback = st.number_input("Lookback Period", value=60, min_value=1)
    with col2:
        horizon = st.number_input("Prediction Horizon", value=5, min_value=1)
    with col3:
        batch_size = st.number_input("Batch Size", value=100, min_value=1)
    
    if st.button("Run Prediction"):
        try:
            # Initialize predictor
            predictor = MarketPredictor(
                lookback_period=lookback,
                prediction_horizon=horizon
            )
            
            # Prepare data
            scaled_data = predictor.scaler.fit_transform(data['close'].values.reshape(-1, 1))
            X, y = predictor.prepare_sequences(scaled_data)
            
            # Build and train model
            predictor.model = predictor.build_model(input_shape=(predictor.lookback_period, 1))
            
            with st.spinner("Training model..."):
                predictor.model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            
            # Get date range for backtesting
            dates = pd.date_range(start='2022-01-01', periods=len(data), freq='D')
            test_start = dates[int(len(dates)*0.7)].strftime('%Y-%m-%d')
            test_end = dates[-1].strftime('%Y-%m-%d')
            
            # Run backtesting
            with st.spinner("Generating predictions..."):
                results = predictor.parallel_backtest(
                    data=pd.Series(data['close'].values, index=dates),
                    start_date=test_start,
                    end_date=test_end,
                    batch_size=batch_size
                )
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{results['rmse']:.4f}")
            col2.metric("MAE", f"{results['mae']:.4f}")
            
            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['actual'][:, 0],
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=results['predictions'][:, 0],
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Market Price Prediction vs Actual",
                xaxis_title="Time",
                yaxis_title="Price",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution
            errors = results['predictions'][:, 0] - results['actual'][:, 0]
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                name='Error Distribution'
            ))
            
            fig_error.update_layout(
                title="Prediction Error Distribution",
                xaxis_title="Error",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your data format and try again")

else:
    st.info("Please upload a CSV file with a 'close' price column to begin prediction")