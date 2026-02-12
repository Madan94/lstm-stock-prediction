"""Configuration constants for the financial forecasting platform."""

# Indices configuration
INDICES = {
    "SP500": {"symbol": "^GSPC", "name": "S&P 500"},
    "DJIA": {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
    "NASDAQ": {"symbol": "^IXIC", "name": "NASDAQ Composite"}
}

# Data configuration (optimized for 1 year of data)
YEARS_OF_DATA = 1  # One year of S&P 500 data
LOOKBACK_WINDOW = 20  # 20-day lookback window (reduced for limited data)
TRAIN_WINDOW_YEARS = 0.7  # ~7 months training window
TEST_WINDOW_MONTHS = 1  # 1 month test window
USE_SIMPLE_SPLIT = True  # Use simple train/test split instead of walk-forward for limited data
TRAIN_SPLIT_RATIO = 0.8  # 80% train, 20% test when using simple split

# Model configuration (optimized for higher accuracy)
LSTM_HIDDEN_SIZE = 256  # Increased for more capacity
LSTM_NUM_LAYERS = 3  # Deeper network
DROPOUT = 0.3  # Slightly higher dropout for regularization
LEARNING_RATE = 0.0005  # Lower learning rate for stability
BATCH_SIZE = 64  # Larger batch size
NUM_EPOCHS = 100  # More epochs with early stopping
WEIGHT_DECAY = 1e-5  # L2 regularization

# Asymmetric loss configuration
UPWARD_PENALTY = 1.5  # α - penalty for missing upward moves
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



