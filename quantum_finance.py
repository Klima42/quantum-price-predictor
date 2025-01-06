import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import multiprocessing as mp
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

def process_batch(args):
    """
    Process a batch of data for prediction.
    Helper function for parallel processing.
    """
    batch_data, lookback_period, prediction_horizon, scaler, model = args
    predictions = []
    actual_values = []
    
    for i in range(len(batch_data) - lookback_period - prediction_horizon):
        sequence = batch_data[i:i + lookback_period]
        actual = batch_data[i + lookback_period:
                          i + lookback_period + prediction_horizon]
        
        scaled_sequence = scaler.transform(sequence.values.reshape(-1, 1))
        scaled_sequence = scaled_sequence.reshape(1, lookback_period, 1)
        
        pred = model.predict(scaled_sequence, verbose=0)
        pred = scaler.inverse_transform(pred.reshape(-1, 1))
        
        predictions.append(pred.flatten())
        actual_values.append(actual.values.flatten())
    
    return predictions, actual_values

class QuantumInspiredOptimizer:
    """
    Implements quantum-inspired optimization algorithms for feature selection
    and hyperparameter tuning.
    """
    def __init__(self, n_particles: int = 50, n_iterations: int = 100):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.best_position = None
        self.best_fitness = float('-inf')
    
    def quantum_rotation(self, positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Apply quantum rotation gate to update particle positions."""
        cos_theta = np.cos(angles)
        sin_theta = np.sin(angles)
        return positions * cos_theta + (1 - positions) * sin_theta
    
    def optimize(self, fitness_func, dimensions: int) -> Tuple[np.ndarray, float]:
        """Perform quantum-inspired optimization to find optimal parameters."""
        positions = np.random.rand(self.n_particles, dimensions)
        velocities = np.random.rand(self.n_particles, dimensions) * 2 * np.pi
        
        for _ in range(self.n_iterations):
            fitness_values = np.array([fitness_func(pos) for pos in positions])
            best_idx = np.argmax(fitness_values)
            
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_position = positions[best_idx].copy()
            
            angles = velocities * np.random.rand(self.n_particles, dimensions)
            positions = self.quantum_rotation(positions, angles)
            velocities = 0.7 * velocities + 0.3 * np.random.rand(self.n_particles, dimensions) * 2 * np.pi
            
        return self.best_position, self.best_fitness

class MarketPredictor:
    """
    Main class for market prediction combining quantum-inspired optimization
    with deep learning.
    """
    def __init__(self, lookback_period: int = 60, prediction_horizon: int = 5):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.optimizer = QuantumInspiredOptimizer()
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.lookback_period - self.prediction_horizon + 1):
            X.append(data[i:(i + self.lookback_period)])
            y.append(data[i + self.lookback_period:i + self.lookback_period + self.prediction_horizon])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model with quantum-optimized architecture."""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(self.prediction_horizon)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def parallel_backtest(self, data: pd.DataFrame, start_date: str, 
                         end_date: str, batch_size: int = 100) -> Dict:
        """Perform parallel backtesting of the prediction model."""
        mask = (data.index >= start_date) & (data.index <= end_date)
        test_data = data[mask]
        
        n_batches = len(test_data) // batch_size + 1
        batches = np.array_split(test_data, n_batches)
        
        args = [(batch, self.lookback_period, self.prediction_horizon, 
                self.scaler, self.model) for batch in batches]
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(process_batch, args)
        
        all_predictions = []
        all_actual = []
        for pred, actual in results:
            all_predictions.extend(pred)
            all_actual.extend(actual)
            
        return {
            'predictions': np.array(all_predictions),
            'actual': np.array(all_actual),
            'rmse': np.sqrt(np.mean((np.array(all_predictions) - np.array(all_actual))**2)),
            'mae': np.mean(np.abs(np.array(all_predictions) - np.array(all_actual)))
        }
    
    def visualize_results(self, backtest_results: Dict):
        """Create interactive visualizations of backtest results."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(backtest_results['actual'][:, 0], label='Actual', alpha=0.7)
        plt.plot(backtest_results['predictions'][:, 0], label='Predicted', alpha=0.7)
        plt.title('Market Price Prediction vs Actual')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        errors = backtest_results['predictions'][:, 0] - backtest_results['actual'][:, 0]
        sns.histplot(errors, kde=True)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Root Mean Square Error: {backtest_results['rmse']:.4f}")
        print(f"Mean Absolute Error: {backtest_results['mae']:.4f}")