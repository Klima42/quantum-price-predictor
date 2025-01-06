import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantum_finance import MarketPredictor

def generate_test_data():
    """Generate synthetic market data for testing."""
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(730)]  # 2 years
    
    np.random.seed(42)  # For reproducibility
    price = 100  # Starting price
    prices = []
    
    for _ in range(730):
        change = np.random.normal(0.0001, 0.02)  # Mean slightly positive for upward trend
        price *= (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    df.to_csv('market_data.csv')
    print("Test data saved to 'market_data.csv'")
    return df

def test_with_custom_data(file_path: str, date_column: str, price_column: str,
                         train_start: str, train_end: str,
                         test_start: str, test_end: str):
    """
    Test the predictor with custom data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing market data
    date_column : str
        Name of the column containing dates
    price_column : str
        Name of the column containing price data
    train_start, train_end : str
        Start and end dates for training data (format: 'YYYY-MM-DD')
    test_start, test_end : str
        Start and end dates for testing data (format: 'YYYY-MM-DD')
    """
    # Load data
    data = pd.read_csv(file_path)
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    
    # Initialize predictor
    predictor = MarketPredictor(lookback_period=60, prediction_horizon=5)
    
    # Prepare training data
    train_data = data[train_start:train_end][price_column]
    
    # Scale the data
    predictor.scaler.fit(train_data.values.reshape(-1, 1))
    
    # Prepare sequences for training
    X_train, y_train = predictor.prepare_sequences(
        predictor.scaler.transform(train_data.values.reshape(-1, 1))
    )
    
    # Build and train the model
    predictor.model = predictor.build_model(input_shape=(X_train.shape[1], 1))
    predictor.model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Perform backtesting
    backtest_results = predictor.parallel_backtest(
        data[price_column],
        start_date=test_start,
        end_date=test_end
    )
    
    # Visualize results
    predictor.visualize_results(backtest_results)

def main():
    try:
        # Example 1: Using synthetic data
        print("Testing with synthetic data...")
        data = generate_test_data()
        
        predictor = MarketPredictor(lookback_period=60, prediction_horizon=5)
        
        # Prepare training data
        train_data = data[:'2022-12-31']
        predictor.scaler.fit(train_data['close'].values.reshape(-1, 1))
        
        X_train, y_train = predictor.prepare_sequences(
            predictor.scaler.transform(train_data['close'].values.reshape(-1, 1))
        )
        
        # Train model
        predictor.model = predictor.build_model(input_shape=(X_train.shape[1], 1))
        predictor.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Test model
        backtest_results = predictor.parallel_backtest(
            data['close'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        predictor.visualize_results(backtest_results)
        
        # Example 2: Using custom data (commented out)
        """
        print("\nTesting with custom data...")
        test_with_custom_data(
            file_path='your_data.csv',
            date_column='Date',
            price_column='Close',
            train_start='2020-01-01',
            train_end='2021-12-31',
            test_start='2022-01-01',
            test_end='2022-12-31'
        )
        """
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()