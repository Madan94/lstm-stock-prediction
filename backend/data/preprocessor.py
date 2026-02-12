"""Data preprocessing and cleaning."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Remove any invalid values in required columns first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Only drop rows where required columns have NaN (not feature columns)
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} rows with missing values in required columns")
    
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


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize features using RobustScaler (less sensitive to outliers).
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        scaler: Pre-fitted scaler (if None, create new)
        fit: Whether to fit the scaler
        
    Returns:
        DataFrame with normalized features, fitted scaler
    """
    if scaler is None:
        scaler = RobustScaler()
    
    df_norm = df.copy()
    
    # Get available feature columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Filter out columns that are all NaN
    valid_cols = []
    for col in available_cols:
        if col in df.columns:
            if not df[col].isna().all():
                valid_cols.append(col)
            else:
                logger.warning(f"Column {col} is all NaN, skipping normalization")
    
    if len(valid_cols) == 0:
        logger.error("No valid feature columns for normalization")
        return df_norm, scaler
    
    # Fill any remaining NaNs with 0 (shouldn't happen if we did ffill/bfill, but safety)
    df_norm[valid_cols] = df_norm[valid_cols].fillna(0)
    
    try:
        if fit:
            df_norm[valid_cols] = scaler.fit_transform(df_norm[valid_cols])
        else:
            df_norm[valid_cols] = scaler.transform(df_norm[valid_cols])
    except Exception as e:
        logger.error(f"Error in normalization: {e}")
        # Fallback: use StandardScaler if RobustScaler fails
        from sklearn.preprocessing import StandardScaler
        fallback_scaler = StandardScaler()
        if fit:
            df_norm[valid_cols] = fallback_scaler.fit_transform(df_norm[valid_cols])
        else:
            df_norm[valid_cols] = fallback_scaler.transform(df_norm[valid_cols])
        scaler = fallback_scaler
    
    return df_norm, scaler


def prepare_sequences(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: list,
    normalize: bool = True,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, Optional[StandardScaler]]:
    """
    Prepare sequences for LSTM training with optional normalization.
    
    Args:
        df: DataFrame with features and target
        lookback: Number of days to look back
        feature_cols: List of feature column names
        normalize: Whether to normalize features
        scaler: Pre-fitted scaler (if None, create new)
        fit_scaler: Whether to fit the scaler
        
    Returns:
        X: Sequences of shape (samples, lookback, features)
        y: Targets of shape (samples,)
        dates: Date index for each sample
        scaler: Fitted scaler (if normalize=True)
    """
    df_work = df.copy()
    
    # Normalize features if requested
    if normalize:
        df_work, scaler = normalize_features(df_work, feature_cols, scaler, fit_scaler)
    
    X, y, dates = [], [], []
    
    # Get available feature columns
    available_feature_cols = [col for col in feature_cols if col in df_work.columns]
    
    if len(available_feature_cols) == 0:
        logger.error(f"No available feature columns in DataFrame. Expected: {feature_cols[:5]}...")
        return np.array([]), np.array([]), pd.DatetimeIndex([]), None
    
    for i in range(lookback, len(df_work)):
        # Extract sequence
        seq = df_work[available_feature_cols].iloc[i-lookback:i].values
        target = df['target'].iloc[i]  # Use original df for target
        date = df.index[i]
        
        # Check for NaN/Inf - fill with 0 if found
        if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        X.append(seq)
        y.append(target)
        dates.append(date)
    
    if len(X) == 0:
        logger.warning(f"No sequences created. DataFrame length: {len(df_work)}, Lookback: {lookback}")
        return np.array([]), np.array([]), pd.DatetimeIndex([]), None
    
    result_scaler = scaler if normalize else None
    return np.array(X), np.array(y), pd.DatetimeIndex(dates), result_scaler







