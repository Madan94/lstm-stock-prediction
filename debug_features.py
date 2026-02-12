import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.data.fetcher import fetch_index_data
from backend.features.indicators import engineer_features, get_feature_columns
from backend.utils.config import INDICES, YEARS_OF_DATA

def debug_features():
    print("Fetching data...")
    symbol = INDICES["SP500"]["symbol"]
    df = fetch_index_data(symbol, years=YEARS_OF_DATA)
    print(f"Original shape: {df.shape}")
    
    print("Engineering features...")
    df_features = engineer_features(df)
    print(f"Post-engineering shape: {df_features.shape}")
    
    # Check for NaNs per column
    nan_counts = df_features.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    print("\nColumns with NaNs:")
    print(cols_with_nans.sort_values(ascending=False).head(20))
    
    total_nas = df_features.isna().any(axis=1).sum()
    print(f"\nTotal rows with at least one NaN: {total_nas} / {len(df_features)}")
    
    # Check specific features if all are NaN
    all_nan_cols = nan_counts[nan_counts == len(df_features)]
    if not all_nan_cols.empty:
        print("\nColumns that are ALL NaN:")
        print(all_nan_cols.index.tolist())

if __name__ == "__main__":
    debug_features()
