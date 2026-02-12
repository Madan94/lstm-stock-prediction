"""Configuration constants for the financial forecasting platform."""

# Indices configuration
INDICES = {
    "SP500": {"symbol": "^GSPC", "name": "S&P 500"},
    "DJI": {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
    "NASDAQ": {"symbol": "^IXIC", "name": "NASDAQ Composite"}
}

# Data configuration
YEARS_OF_DATA = 30  # Fetch 30 years of historical data
LOOKBACK_WINDOW = 150  # Increased from 120 to 150 days for more context
TRAIN_WINDOW_YEARS = 2
TEST_WINDOW_MONTHS = 12

# Model configuration - Optimized for speed and performance
LSTM_HIDDEN_SIZE = 128  # Reduced from 512 for speed (sufficient for this data)
LSTM_NUM_LAYERS = 2  # Reduced from 4 for speed
DROPOUT = 0.2  # Reduced slightly
LEARNING_RATE = 0.001  # Increased slightly for faster convergence
BATCH_SIZE = 64  # Increased from 12 for faster GPU training
NUM_EPOCHS = 50  # Reduced from 150 (early stopping usually triggers sooner)
EARLY_STOPPING_PATIENCE = 5  # Reduced from 10
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping threshold

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by half
LR_SCHEDULER_PATIENCE = 5  # After 5 epochs without improvement
LR_SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate

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




