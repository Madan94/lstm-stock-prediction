"""Walk-forward training and validation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_walk_forward_windows(
    df: pd.DataFrame,
    train_years: float = 2.0,
    test_months: int = 1
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create walk-forward training and testing windows.
    
    Args:
        df: DataFrame with datetime index
        train_years: Years of data for training (can be fractional, e.g., 0.6 for ~6 months)
        test_months: Months of data for testing
        
    Returns:
        List of (train_end_date, test_end_date) tuples
    """
    windows = []
    
    # Check if DataFrame is empty
    if len(df) == 0:
        logger.warning("Empty DataFrame provided to create_walk_forward_windows")
        return windows
    
    start_date = df.index[0]
    end_date = df.index[-1]
    
    # Handle fractional years by converting to days
    train_days = int(train_years * 365.25)  # Account for leap years
    current_date = start_date + pd.DateOffset(days=train_days)
    
    # Ensure we have enough data for lookback window + test period
    min_test_date = current_date + pd.DateOffset(days=60)  # At least 60 days for lookback + some test data
    
    while current_date + pd.DateOffset(months=test_months) <= end_date and min_test_date <= end_date:
        train_end = current_date
        test_end = current_date + pd.DateOffset(months=test_months)
        
        # Only add window if test_end doesn't exceed available data
        if test_end <= end_date:
            windows.append((train_end, test_end))
        
        # Slide forward by test window
        current_date = test_end
        min_test_date = current_date + pd.DateOffset(days=60)
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    return windows


def get_walk_forward_data(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback: int,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Extract training and testing data for a walk-forward window.
    
    Args:
        df: Full DataFrame with features and target
        train_end: End date for training data
        test_end: End date for testing data
        lookback: Lookback window size
        normalize: Whether to normalize features
        
    Returns:
        X_train, y_train, X_test, y_test, train_dates, test_dates
    """
    from backend.data.preprocessor import prepare_sequences
    from backend.features.indicators import get_feature_columns
    
    # Split data
    train_df = df[df.index <= train_end].copy()
    test_df = df[(df.index > train_end) & (df.index <= test_end)].copy()
    
    # Need at least lookback days for training, and some test data
    if len(train_df) < lookback or len(test_df) < 1:
        return None, None, None, None, None, None
    
    feature_cols = get_feature_columns()
    
    # Prepare sequences for training (fit scaler on training data)
    X_train, y_train, train_dates, scaler = prepare_sequences(
        train_df, lookback, feature_cols, normalize=normalize, fit_scaler=True
    )
    
    # For test data, we need to use the last 'lookback' days from training + test period
    # to create sequences that start in the test period
    # Use the scaler fitted on training data
    combined_for_test = pd.concat([train_df.iloc[-lookback:], test_df])
    X_test, y_test, test_dates, _ = prepare_sequences(
        combined_for_test, lookback, feature_cols, 
        normalize=normalize, scaler=scaler, fit_scaler=False
    )
    
    # Filter test sequences to only those that start in the test period
    if len(test_dates) > 0:
        test_mask = test_dates > train_end
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        test_dates = test_dates[test_mask]
    
    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None, None, None
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates


def get_simple_split_data(
    df: pd.DataFrame,
    lookback: int,
    train_ratio: float = 0.8,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Simple train/test split for limited data scenarios.
    
    Args:
        df: Full DataFrame with features and target
        lookback: Lookback window size
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
        normalize: Whether to normalize features
        
    Returns:
        X_train, y_train, X_test, y_test, train_dates, test_dates
    """
    from backend.data.preprocessor import prepare_sequences
    from backend.features.indicators import get_feature_columns
    
    if len(df) < lookback + 10:
        logger.warning(f"Not enough data for simple split: {len(df)} rows, need at least {lookback + 10}")
        return None, None, None, None, None, None
    
    feature_cols = get_feature_columns()
    
    # Prepare all sequences first
    X_all, y_all, dates_all, scaler = prepare_sequences(
        df, lookback, feature_cols, normalize=normalize, fit_scaler=True
    )
    
    if len(X_all) == 0:
        logger.warning(f"No sequences created. DataFrame length: {len(df)}, Lookback: {lookback}")
        logger.warning(f"Available feature columns: {[col for col in feature_cols if col in df.columns][:10]}")
        return None, None, None, None, None, None
    
    logger.info(f"Created {len(X_all)} sequences from {len(df)} rows (lookback: {lookback})")
    
    # Split based on ratio
    split_idx = int(len(X_all) * train_ratio)
    
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    train_dates = dates_all[:split_idx]
    
    X_test = X_all[split_idx:]
    y_test = y_all[split_idx:]
    test_dates = dates_all[split_idx:]
    
    logger.info(f"Simple split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None, None, None
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates


def walk_forward_predict(
    model,
    X_test: np.ndarray,
    device: str = 'cpu',
    return_attention: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions using walk-forward approach.
    
    Args:
        model: Trained model
        X_test: Test sequences
        device: Device to run on
        return_attention: Whether to return attention weights
        
    Returns:
        predictions: Predicted probabilities
        attention_weights: Attention weights if requested
    """
    import torch
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        if return_attention:
            output, attention = model(X_test_t, return_attention=True)
            return output.cpu().numpy(), attention.cpu().numpy()
        else:
            output = model(X_test_t)
            return output.cpu().numpy(), None



