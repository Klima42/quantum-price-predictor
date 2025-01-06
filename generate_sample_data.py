import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample dates
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate synthetic price data with realistic patterns
np.random.seed(42)  # For reproducibility

# Start with a base price
base_price = 100
n_days = len(dates)

# Generate random walks with trends and seasonality
random_walk = np.random.normal(0.001, 0.02, n_days).cumsum()
trend = np.linspace(0, 0.5, n_days)  # Upward trend
seasonality = 0.1 * np.sin(np.linspace(0, 8*np.pi, n_days))  # Seasonal pattern

# Combine components
price_data = base_price * (1 + random_walk + trend + seasonality)

# Add some volatility clusters
volatility = np.random.normal(1, 0.02, n_days)
price_data = price_data * volatility

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'open': price_data * (1 + np.random.normal(0, 0.002, n_days)),
    'high': price_data * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
    'low': price_data * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
    'close': price_data,
    'volume': np.random.normal(1000000, 200000, n_days).astype(int)
})

# Save to CSV
df.to_csv('sample_market_data.csv', index=False)
print("Sample market data has been generated and saved to 'sample_market_data.csv'")