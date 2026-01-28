"""Data fetching module using yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def fetch_index_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Fetch historical data for an index.
    
    Args:
        symbol: Index symbol (e.g., '^GSPC')
        years: Number of years of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            raise ValueError(f"No data retrieved for {symbol}")
        
        # Ensure proper column names
        if hasattr(df, 'columns'):
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        else:
            raise ValueError(f"Invalid data format for {symbol}")
        
        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Sort by date
        df = df.sort_index()
        
        logger.info(f"Fetched {len(df)} days of data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise




def get_all_indices_data(years: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all configured indices.
    
    Args:
        years: Number of years of historical data
        
    Returns:
        Dictionary mapping index names to DataFrames
    """
    from backend.utils.config import INDICES
    
    data = {}
    for index_name, config in INDICES.items():
        try:
            data[index_name] = fetch_index_data(config["symbol"], years)
        except Exception as e:
            logger.warning(f"Failed to fetch {index_name}: {e}")
    
    return data



