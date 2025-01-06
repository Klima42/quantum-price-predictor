import pytest
import numpy as np
import pandas as pd
from quantum_market_predictor.core.market_predictor import (
    MarketPredictor,
    QuantumInspiredOptimizer,
    process_batch
)

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    return pd.Series(prices, index=dates, name='price')

@pytest.fixture
def predictor():
    """Create a MarketPredictor instance for testing."""
    return MarketPredictor(lookback_period=30, prediction_horizon=5)

def test_quantum_optimizer_initialization():
    """Test QuantumInspiredOptimizer initialization."""
    optimizer = QuantumInspiredOptimizer(n_particles=40, n_iterations=80)
    assert optimizer.n_particles == 40
    assert optimizer.n_iterations == 80
    assert optimizer.best_position is None
    assert optimizer.best_fitness == float('-inf')

def test_quantum_rotation():
    """Test quantum rotation operation."""
    optimizer = QuantumInspiredOptimizer()
    positions = np.array([[0.5, 0.3], [0.7, 0.4]])
    angles = np.array([[np.pi/4, np.pi/6], [np.pi/3, np.pi/4]])
    
    rotated = optimizer.quantum_rotation(positions, angles)
    assert rotated.shape == positions.shape
    assert np.all(rotated >= 0) and np.all(rotated <= 1)

def test_optimizer_optimize():
    """Test optimization process."""
    optimizer = QuantumInspiredOptimizer(n_iterations=10)
    
    # Simple quadratic function to minimize
    def fitness_func(x):
        return -(x[0]**2 + x[1]**2)  # Negative because optimizer maximizes
    
    best_position, best_fitness = optimizer.optimize(fitness_func, dimensions=2)
    assert len(best_position) == 2
    assert isinstance(best_fitness, float)

def test_market_predictor_initialization(predictor):
    """Test MarketPredictor initialization."""
    assert predictor.lookback_period == 30
    assert predictor.prediction_horizon == 5
    assert predictor.model is None
    assert isinstance(predictor.optimizer, QuantumInspiredOptimizer)

def test_prepare_sequences(predictor, sample_data):
    """Test sequence preparation for LSTM."""
    scaled_data = predictor.scaler.fit_transform(sample_data.values.reshape(-1, 1))
    X, y = predictor.prepare_sequences(scaled_data)
    
    expected_x_shape = (
        len(scaled_data) - predictor.lookback_period - predictor.prediction_horizon + 1,
        predictor.lookback_period
    )
    expected_y_shape = (
        len(scaled_data) - predictor.lookback_period - predictor.prediction_horizon + 1,
        predictor.prediction_horizon
    )
    
    assert X.shape == expected_x_shape
    assert y.shape == expected_y_shape

def test_build_model(predictor):
    """Test LSTM model building."""
    model = predictor.build_model(input_shape=(30, 1))
    
    assert isinstance(model.layers[0], type(tf.keras.layers.LSTM()))
    assert model.layers[0].units == 128
    assert isinstance(model.layers[-1], type(tf.keras.layers.Dense()))
    assert model.layers[-1].units == predictor.prediction_horizon

def test_process_batch(predictor, sample_data):
    """Test batch processing for parallel backtesting."""
    # Prepare test data
    predictor.model = predictor.build_model(input_shape=(predictor.lookback_period, 1))
    scaled_data = predictor.scaler.fit_transform(sample_data.values.reshape(-1, 1))
    batch = sample_data[:100]
    
    # Test batch processing
    predictions, actual_values = process_batch(
        (batch, predictor.lookback_period, predictor.prediction_horizon,
         predictor.scaler, predictor.model)
    )
    
    assert len(predictions) > 0
    assert len(actual_values) > 0
    assert len(predictions) == len(actual_values)
    assert predictions[0].shape == (predictor.prediction_horizon,)

@pytest.mark.integration
def test_parallel_backtest(predictor, sample_data):
    """Test parallel backtesting functionality."""
    # Prepare model
    predictor.model = predictor.build_model(input_shape=(predictor.lookback_period, 1))
    scaled_data = predictor.scaler.fit_transform(sample_data.values.reshape(-1, 1))
    X, y = predictor.prepare_sequences(scaled_data)
    predictor.model.fit(X, y, epochs=1, verbose=0)
    
    # Perform backtesting
    results = predictor.parallel_backtest(
        data=sample_data,
        start_date='2022-06-01',
        end_date='2022-12-31',
        batch_size=50
    )
    
    assert 'predictions' in results
    assert 'actual' in results
    assert 'rmse' in results
    assert 'mae' in results
    assert isinstance(results['rmse'], float)
    assert isinstance(results['mae'], float)
    assert len(results['predictions']) > 0
    assert len(results['actual']) > 0

@pytest.mark.integration
def test_end_to_end_workflow(predictor, sample_data):
    """Test complete workflow from data preparation to prediction."""
    # Prepare data
    scaled_data = predictor.scaler.fit_transform(sample_data.values.reshape(-1, 1))
    X, y = predictor.prepare_sequences(scaled_data)
    
    # Build and train model
    predictor.model = predictor.build_model(input_shape=(predictor.lookback_period, 1))
    history = predictor.model.fit(
        X, y,
        epochs=2,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Perform backtesting
    results = predictor.parallel_backtest(
        data=sample_data,
        start_date='2022-06-01',
        end_date='2022-12-31',
        batch_size=50
    )
    
    assert history.history['loss'][-1] < history.history['loss'][0]
    assert results['rmse'] > 0
    assert results['mae'] > 0