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


def calculate_stochastic(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    
    # %K
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    # Smooth %K
    k = k.rolling(window=smooth_k).mean()
    # %D (signal line)
    d = k.rolling(window=smooth_d).mean()
    
    return pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    })


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (volatility measure)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
    return williams_r


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3  # Typical price
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3  # Typical price
    mf = tp * df['volume']  # Money flow
    
    # Positive and negative money flow
    mf_sign = np.sign(tp.diff())
    positive_mf = mf.where(mf_sign > 0, 0).rolling(window=period).sum()
    negative_mf = mf.where(mf_sign < 0, 0).abs().rolling(window=period).sum()
    
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price ratio."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap = (tp * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    vwap_ratio = df['close'] / vwap
    
    return vwap_ratio


def calculate_momentum(prices: pd.Series, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """Calculate price momentum over multiple periods."""
    momentum_df = pd.DataFrame()
    
    for period in periods:
        momentum_df[f'momentum_{period}'] = prices.pct_change(period)
    
    return momentum_df


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns."""
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=period).std()


def calculate_fractal_dimension(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate fractal dimension (Hurst exponent) for trend strength."""
    def hurst(ts):
        """Calculate Hurst exponent."""
        ts = np.array(ts)
        if len(ts) < 20:
            return 1.5  # Default to random walk
            
        lags = range(2, min(20, len(ts) // 2))
        
        # Calculate tau with handling for zero variance
        tau = []
        for lag in lags:
            diff = np.subtract(ts[lag:], ts[:-lag])
            std = np.std(diff)
            if std == 0:
                std = 1e-10  # Avoid logOfZero
            tau.append(std)
            
        # Fit line
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 1.5  # Return random walk value on failure
    
    return prices.rolling(window=window).apply(hurst, raw=True)


def calculate_entropy(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Shannon entropy of price changes."""
    def entropy(x):
        if len(x) < 2:
            return 0
        x = pd.Series(x)
        x = x[x != 0]
        if len(x) == 0:
            return 0
        value, counts = np.unique(np.sign(x.diff().dropna()), return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    return prices.rolling(window=window).apply(entropy, raw=False)


def calculate_autocorrelation(prices: pd.Series, lag: int = 1, window: int = 20) -> pd.Series:
    """Calculate rolling autocorrelation."""
    return prices.rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0,
        raw=False
    )


def detect_market_regime(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Detect market regime: trending, ranging, volatile."""
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    
    # ADX-like calculation for trend strength
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Trend strength
    directional_move = df['close'] - df['close'].shift(window)
    trend_strength = np.abs(directional_move) / (atr * window)
    
    # Regime encoding: 0=ranging, 1=trending, 2=volatile
    regime = pd.Series(0, index=df.index)
    regime[trend_strength > trend_strength.rolling(window=252).quantile(0.7)] = 1
    regime[volatility > volatility.rolling(window=252).quantile(0.8)] = 2
    
    return regime


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features from raw OHLCV data with advanced analytics.
    
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
    df['rsi_sma'] = df['rsi'].rolling(window=14).mean()
    df['rsi_divergence'] = df['rsi'] - df['rsi_sma']
    
    # Multiple RSI timeframes
    df['rsi_7'] = calculate_rsi(close, 7)
    df['rsi_21'] = calculate_rsi(close, 21)
    
    # MACD
    macd_df = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df = pd.concat([df, macd_df], axis=1)
    df['macd_momentum'] = df['macd_histogram'].diff()
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    df = pd.concat([df, bb_df], axis=1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)).astype(int)
    
    # Stochastic Oscillator
    stoch_df = calculate_stochastic(df)
    df = pd.concat([df, stoch_df], axis=1)
    df['stoch_divergence'] = df['stoch_k'] - df['stoch_d']
    
    # ATR (volatility)
    df['atr'] = calculate_atr(df)
    df['atr_ratio'] = df['atr'] / close
    df['atr_percentile'] = df['atr'].rolling(window=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    
    # OBV (volume flow)
    df['obv'] = calculate_obv(df)
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / df['obv_ma']
    df['obv_momentum'] = df['obv'].pct_change(5)
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(df)
    
    # CCI
    df['cci'] = calculate_cci(df)
    df['cci_sma'] = df['cci'].rolling(window=20).mean()
    
    # MFI
    df['mfi'] = calculate_mfi(df)
    
    # VWAP
    df['vwap_ratio'] = calculate_vwap(df)
    
    # Momentum indicators (multiple timeframes)
    momentum_df = calculate_momentum(close, periods=[3, 5, 10, 20, 50])
    df = pd.concat([df, momentum_df], axis=1)
    
    # Momentum acceleration
    df['momentum_accel_5'] = df['momentum_5'].diff()
    df['momentum_accel_20'] = df['momentum_20'].diff()
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_volatility'] = df['volume'].rolling(window=20).std() / df['volume_ma']
    df['volume_trend'] = df['volume'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # Price action features
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    
    # Moving averages
    df['sma_5'] = close.rolling(window=5).mean()
    df['sma_10'] = close.rolling(window=10).mean()
    df['sma_20'] = close.rolling(window=20).mean()
    df['sma_50'] = close.rolling(window=50).mean()
    df['sma_200'] = close.rolling(window=200).mean()
    
    df['price_sma5_ratio'] = close / df['sma_5']
    df['price_sma10_ratio'] = close / df['sma_10']
    df['price_sma20_ratio'] = close / df['sma_20']
    df['price_sma50_ratio'] = close / df['sma_50']
    df['price_sma200_ratio'] = close / df['sma_200']
    
    # MA crossovers
    df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
    df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
    df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # EMA
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    df['price_ema12_ratio'] = close / df['ema_12']
    df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
    
    # Trend indicators
    df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    
    # Volatility regimes
    df['volatility_percentile'] = df['volatility'].rolling(window=252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    
    # Advanced features for maximum accuracy
    df['fractal_dimension'] = calculate_fractal_dimension(close, window=20)
    df['price_entropy'] = calculate_entropy(close, window=20)
    df['autocorr_1'] = calculate_autocorrelation(close, lag=1, window=20)
    df['autocorr_5'] = calculate_autocorrelation(close, lag=5, window=20)
    
    # Market regime detection
    df['market_regime'] = detect_market_regime(df, window=60)
    
    # Feature interactions (important for non-linear patterns)
    df['rsi_momentum'] = df['rsi'] * df['momentum_10']
    df['volatility_volume'] = df['volatility'] * df['volume_ratio']
    df['trend_strength'] = df['momentum_20'] * (1 - df['volatility_percentile'])
    
    # Distance from key levels
    df['dist_from_high_20'] = (close - df['high'].rolling(20).max()) / close
    df['dist_from_low_20'] = (close - df['low'].rolling(20).min()) / close
    df['dist_from_high_60'] = (close - df['high'].rolling(60).max()) / close
    df['dist_from_low_60'] = (close - df['low'].rolling(60).min()) / close
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get list of feature column names for model input.
    
    Returns:
        List of feature column names
    """
    return [
        # Price returns and volatility
        'returns', 'log_returns', 'volatility', 'volatility_percentile',
        # RSI (multiple timeframes)
        'rsi', 'rsi_sma', 'rsi_divergence', 'rsi_7', 'rsi_21',
        # MACD
        'macd', 'macd_signal', 'macd_histogram', 'macd_momentum',
        # Bollinger Bands
        'bb_width', 'bb_position', 'bb_squeeze',
        # Stochastic
        'stoch_k', 'stoch_d', 'stoch_divergence',
        # ATR
        'atr_ratio', 'atr_percentile',
        # OBV
        'obv_ratio', 'obv_momentum',
        # Williams %R
        'williams_r',
        # CCI
        'cci', 'cci_sma',
        # MFI
        'mfi',
        # VWAP
        'vwap_ratio',
        # Momentum (multiple timeframes)
        'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
        'momentum_accel_5', 'momentum_accel_20',
        # Volume
        'volume_ratio', 'volume_volatility', 'volume_trend',
        # Price action
        'high_low_ratio', 'close_position', 'body_size', 'upper_shadow', 'lower_shadow',
        # Moving average ratios
        'price_sma5_ratio', 'price_sma10_ratio', 'price_sma20_ratio', 
        'price_sma50_ratio', 'price_sma200_ratio',
        'price_ema12_ratio',
        # Crossovers
        'sma_5_10_cross', 'sma_10_20_cross', 'sma_20_50_cross', 'sma_50_200_cross',
        'sma_cross', 'ema_cross',
        # Advanced features
        'fractal_dimension', 'price_entropy', 'autocorr_1', 'autocorr_5',
        # Market regime
        'market_regime',
        # Feature interactions
        'rsi_momentum', 'volatility_volume', 'trend_strength',
        # Distance from levels
        'dist_from_high_20', 'dist_from_low_20', 'dist_from_high_60', 'dist_from_low_60'
    ]






