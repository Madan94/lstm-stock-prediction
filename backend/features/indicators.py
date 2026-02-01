"""Technical indicators and feature engineering."""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Band width and position
    band_width = (upper_band - lower_band) / sma
    band_position = (prices - lower_band) / (upper_band - lower_band)
    
    return pd.DataFrame({
        'bb_upper': upper_band,
        'bb_middle': sma,
        'bb_lower': lower_band,
        'bb_width': band_width,
        'bb_position': band_position
    })


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns."""
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=period).std()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features from raw OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with engineered features
    """
    from backend.utils.config import (
        RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
        BB_PERIOD, BB_STD, VOLATILITY_PERIOD
    )
    
    df = df.copy()
    close = df['close']
    
    # Price-based features
    df['returns'] = calculate_returns(close)
    df['log_returns'] = calculate_log_returns(close)
    df['volatility'] = calculate_volatility(df['returns'], VOLATILITY_PERIOD)
    
    # RSI
    df['rsi'] = calculate_rsi(close, RSI_PERIOD)
    
    # MACD
    macd_df = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df = pd.concat([df, macd_df], axis=1)
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    df = pd.concat([df, bb_df], axis=1)
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price position features
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Moving averages
    df['sma_20'] = close.rolling(window=20).mean()
    df['sma_50'] = close.rolling(window=50).mean()
    df['price_sma20_ratio'] = close / df['sma_20']
    df['price_sma50_ratio'] = close / df['sma_50']
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names for model input.
    
    Returns:
        List of feature column names
    """
    return [
        'returns', 'log_returns', 'volatility',
        'rsi',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_width', 'bb_position',
        'volume_ratio',
        'high_low_ratio', 'close_position',
        'price_sma20_ratio', 'price_sma50_ratio'
    ]





