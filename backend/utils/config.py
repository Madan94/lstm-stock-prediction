"""Configuration constants for the financial forecasting platform."""

# Indices configuration
INDICES = {
    "SP500": {"symbol": "^GSPC", "name": "S&P 500"},
    "DJI": {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
    "NASDAQ": {"symbol": "^IXIC", "name": "NASDAQ Composite"}
}

# Data configuration
YEARS_OF_DATA = 30  # Fetch 30 years of historical data
LOOKBACK_WINDOW = 60  # days
TRAIN_WINDOW_YEARS = 2
TEST_WINDOW_MONTHS = 1

# Model configuration
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 16  # Reduced from 32 to manage GPU memory with large datasets
NUM_EPOCHS = 50

# Asymmetric loss configuration
UPWARD_PENALTY = 2.0  # α - penalty for missing upward moves
DOWNWARD_PENALTY = 1.0  # β - penalty for false positives

# Backtesting configuration
TRANSACTION_COST = 0.001  # 0.1% per trade
TRADING_DAYS_PER_YEAR = 252

# Technical indicators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
VOLATILITY_PERIOD = 20

# Paths
MODELS_DIR = "backend/models/saved"
RESULTS_DIR = "backend/data/results"



