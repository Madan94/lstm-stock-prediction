#!/usr/bin/env python
"""Script to import 30 years of SP500 data."""

import sys
import os
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from data.fetcher import fetch_index_data
from utils.config import INDICES

def _load_existing(results_dir: Path, index_name: str) -> pd.DataFrame | None:
    pickle_path = results_dir / f"{index_name}_data.pkl"
    csv_path = results_dir / f"{index_name}_data.csv"

    if pickle_path.exists():
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None


def _merge_existing(existing: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df
    df = pd.concat([existing, new_df])
    df = df[~df.index.duplicated(keep='first')]
    return df.sort_index()


def main():
    index_name = "SP500"
    print(f"Importing 30 years for {index_name}...")
    
    results_dir = Path("backend/data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        symbol = INDICES[index_name]["symbol"]
        new_df = fetch_index_data(symbol, years=30)
        
        existing_df = _load_existing(results_dir, index_name)
        merged_df = _merge_existing(existing_df, new_df)
        
        pickle_path = results_dir / f"{index_name}_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(merged_df, f)
        print(f"✓ Saved {index_name}: {len(merged_df)} days ({merged_df.index[0].date()} to {merged_df.index[-1].date()})")
        
        csv_path = results_dir / f"{index_name}_data.csv"
        merged_df.to_csv(csv_path)
        print(f"  CSV saved to: {csv_path}")
    except Exception as e:
        print(f"✗ Failed to fetch {index_name}: {e}")

if __name__ == "__main__":
    main()
