
#!/usr/bin/env python
"""Script to import historical data for S&P 500, DJI, and NASDAQ."""

import sys
import os
import pickle
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from data.fetcher import get_all_indices_data, fetch_index_data
from utils.config import INDICES, YEARS_OF_DATA

SP500_YEARS = 30

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
    """Import and save data for all indices."""
    print("Importing data for indices...")
    
    # Define durations
    durations = {
        "SP500": 30,
        "NASDAQ": 23,
        "DJI": 23
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path("backend/data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for index_name, years in durations.items():
        if index_name not in INDICES:
            print(f"Skipping unknown index: {index_name}")
            continue
            
        print(f"\nFetching {years} years for {index_name}...")
        try:
            symbol = INDICES[index_name]["symbol"]
            df = fetch_index_data(symbol, years=years)
            
            if df is not None and not df.empty:
                existing_df = _load_existing(results_dir, index_name)
                df = _merge_existing(existing_df, df)

                # Save as pickle
                pickle_path = results_dir / f"{index_name}_data.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(df, f)
                print(f"Saved {index_name}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
                
                # Also save as CSV for inspection
                csv_path = results_dir / f"{index_name}_data.csv"
                df.to_csv(csv_path)
                print(f"  CSV saved to: {csv_path}")
            else:
                print(f"Failed to fetch {index_name}")
        except Exception as e:
            print(f"Failed to fetch {index_name}: {e}")
    
    print("\nData import complete!")

if __name__ == "__main__":
    main()
