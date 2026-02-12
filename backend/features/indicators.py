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


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smooth TR and DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'stoch_k': k_percent,
        'stoch_d': d_percent
    })


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum indicator."""
    return close.diff(period)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr


def calculate_market_regime(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Identify market regime: 1 = uptrend, 0 = downtrend, 0.5 = sideways."""
    close = df['close']
    sma_short = close.rolling(window=20).mean()
    sma_long = close.rolling(window=period).mean()
    
    # Trend strength
    trend = (sma_short > sma_long).astype(float)
    
    # Volatility regime
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std()
    vol_percentile = volatility.rolling(window=252).rank(pct=True)
    
    # Combine: strong trend + low volatility = clear regime
    regime = trend.copy()
    regime[(vol_percentile > 0.7) & (abs(trend - 0.5) < 0.3)] = 0.5  # High vol = uncertain
    
    return regime


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
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price-based features
    df['returns'] = calculate_returns(close)
    df['log_returns'] = calculate_log_returns(close)
    df['volatility'] = calculate_volatility(df['returns'], VOLATILITY_PERIOD)
    
    # Momentum features
    df['momentum_10'] = calculate_momentum(close, 10)
    df['momentum_20'] = calculate_momentum(close, 20)
    df['momentum_50'] = calculate_momentum(close, 50)
    
    # RSI (multiple periods for robustness)
    df['rsi'] = calculate_rsi(close, RSI_PERIOD)
    df['rsi_7'] = calculate_rsi(close, 7)
    df['rsi_21'] = calculate_rsi(close, 21)
    
    # MACD
    macd_df = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df = pd.concat([df, macd_df], axis=1)
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    df = pd.concat([df, bb_df], axis=1)
    
    # ADX (trend strength)
    df['adx'] = calculate_adx(high, low, close, 14)
    
    # Stochastic Oscillator
    stoch_df = calculate_stochastic(high, low, close, 14, 3)
    df = pd.concat([df, stoch_df], axis=1)
    
    # CCI
    df['cci'] = calculate_cci(high, low, close, 20)
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(high, low, close, 14)
    
    # Volume features
    df['volume_ma'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma']
    df['obv'] = calculate_obv(close, volume)
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / df['obv_ma']
    
    # Price position features
    df['high_low_ratio'] = high / low
    df['close_position'] = (close - low) / (high - low)
    
    # Moving averages (multiple periods)
    df['sma_10'] = close.rolling(window=10).mean()
    df['sma_20'] = close.rolling(window=20).mean()
    df['sma_50'] = close.rolling(window=50).mean()
    df['sma_200'] = close.rolling(window=200).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    
    # Price ratios
    df['price_sma10_ratio'] = close / df['sma_10']
    df['price_sma20_ratio'] = close / df['sma_20']
    df['price_sma50_ratio'] = close / df['sma_50']
    df['price_sma200_ratio'] = close / df['sma_200']
    
    # MA crossovers
    df['sma_cross_10_20'] = (df['sma_10'] > df['sma_20']).astype(float)
    df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(float)
    df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(float)
    
    # Market regime
    df['market_regime'] = calculate_market_regime(df, 50)
    
    # Volatility features (use smaller window for limited data)
    volatility_mean = df['volatility'].rolling(window=min(60, len(df))).mean()
    df['volatility_ratio'] = df['volatility'] / volatility_mean.replace(0, np.nan)
    df['atr'] = (high - low).rolling(window=14).mean()
    df['atr_ratio'] = df['atr'] / close
    
    # Price patterns
    df['higher_high'] = ((high > high.shift(1)) & (high.shift(1) > high.shift(2))).astype(float)
    df['lower_low'] = ((low < low.shift(1)) & (low.shift(1) < low.shift(2))).astype(float)
    
    # Rate of change
    df['roc_10'] = close.pct_change(10) * 100
    df['roc_20'] = close.pct_change(20) * 100
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names for model input.
    
    Returns:
        List of feature column names
    """
    return [
        # Returns and volatility
        'returns', 'log_returns', 'volatility', 'volatility_ratio',
        # Momentum
        'momentum_10', 'momentum_20', 'momentum_50',
        # RSI (multiple periods)
        'rsi', 'rsi_7', 'rsi_21',
        # MACD
        'macd', 'macd_signal', 'macd_histogram',
        # Bollinger Bands
        'bb_width', 'bb_position',
        # ADX
        'adx',
        # Stochastic
        'stoch_k', 'stoch_d',
        # CCI
        'cci',
        # Williams %R
        'williams_r',
        # Volume
        'volume_ratio', 'obv', 'obv_ma', 'obv_ratio',
        # Price position
        'high_low_ratio', 'close_position',
        # Moving averages
        'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        # Price ratios
        'price_sma10_ratio', 'price_sma20_ratio', 'price_sma50_ratio', 'price_sma200_ratio',
        # MA crossovers
        'sma_cross_10_20', 'sma_cross_20_50', 'ema_cross',
        # Market regime
        'market_regime',
        # ATR
        'atr', 'atr_ratio',
        # Price patterns
        'higher_high', 'lower_low',
        # Rate of change
        'roc_10', 'roc_20'
    ]







