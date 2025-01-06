# Quantum-Inspired Market Predictor

A sophisticated market prediction system that combines quantum-inspired optimization with LSTM neural networks. Features a web interface for easy interaction and visualization.

## Features

- **Quantum-Inspired Optimization**: Leverages quantum computing concepts for parameter optimization
- **LSTM Neural Networks**: Deep learning model for time series prediction
- **Parallel Processing**: Efficient backtesting through parallel computation
- **Interactive Web Interface**: Easy-to-use Streamlit interface for data upload and visualization
- **Real-time Training Visualization**: Monitor model training progress and metrics
- **Comprehensive Error Analysis**: Detailed prediction error analysis and visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-price-predictor.git
cd quantum-price-predictor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the web application:
```bash
streamlit run web_app/app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload your market data CSV file (should include a 'close' price column)

4. Adjust model parameters in the sidebar if desired

5. Click "Run Prediction" to start training and prediction

### Generate Sample Data

To create sample market data for testing:

```bash
python generate_sample_data.py
```

This will create a `sample_market_data.csv` file with synthetic market data.

## Project Structure

```
quantum-price-predictor/
├── quantum_predictor/
│   ├── __init__.py
│   └── predictor.py        # Core prediction engine
├── web_app/
│   ├── __init__.py
│   └── app.py             # Streamlit web interface
├── generate_sample_data.py # Data generation script
└── requirements.txt       # Project dependencies
```

## Model Parameters

- **Lookback Period**: Number of past days to consider for prediction
- **Prediction Horizon**: Number of future days to predict
- **LSTM Architecture**:
  - First layer units: 128 (default)
  - Second layer units: 64 (default)
  - Dropout rate: 0.2 (default)
- **Quantum Optimizer**:
  - Number of particles: 50 (default)
  - Number of iterations: 100 (default)

## Input Data Format

The CSV file should contain at least:
- A 'close' price column
- Daily market data
- Preferably OHLCV format (Open, High, Low, Close, Volume)

Example:
```
date,open,high,low,close,volume
2022-01-01,100.0,101.5,99.0,100.5,1000000
2022-01-02,100.5,102.0,100.0,101.2,1200000
...
```

## Performance Metrics

The system provides several metrics for evaluation:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Error distribution visualization
- Actual vs Predicted price comparison

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with TensorFlow and Streamlit
- Inspired by quantum computing concepts
- Optimized for parallel processing
