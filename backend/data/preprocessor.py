"""Data preprocessing and cleaning."""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for feature engineering.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove rows with missing values
    initial_len = len(df)
    df = df.dropna()
    
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} rows with missing values")
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Remove any remaining invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary directional target (1 = up, 0 = down).
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with 'target' column added
    """
    df = df.copy()
    
    # Calculate next day return
    df['next_return'] = df['close'].pct_change().shift(-1)
    
    # Binary target: 1 if next return > 0, else 0
    df['target'] = (df['next_return'] > 0).astype(int)
    
    # Remove last row (no next day return)
    df = df.iloc[:-1]
    
    return df


def prepare_sequences(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: list
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prepare sequences for LSTM training.
    
    Args:
        df: DataFrame with features and target
        lookback: Number of days to look back
        feature_cols: List of feature column names
        
    Returns:
        X: Sequences of shape (samples, lookback, features)
        y: Targets of shape (samples,)
        dates: Date index for each sample
    """
    X, y, dates = [], [], []
    
    for i in range(lookback, len(df)):
        # Extract sequence
        seq = df[feature_cols].iloc[i-lookback:i].values
        target = df['target'].iloc[i]
        date = df.index[i]
        
        X.append(seq)
        y.append(target)
        dates.append(date)
    
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)





