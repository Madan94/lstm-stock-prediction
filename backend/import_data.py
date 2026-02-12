import os
import sys
import pickle
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.fetcher import fetch_index_data
from backend.utils.config import INDICES, YEARS_OF_DATA, RESULTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fetch and save data for all configured indices."""
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info(f"Starting data import for {YEARS_OF_DATA} years...")
    
    for index_name, config in INDICES.items():
        symbol = config["symbol"]
        logger.info(f"Fetching data for {index_name} ({symbol})...")
        
        try:
            # Fetch data
            df = fetch_index_data(symbol, years=YEARS_OF_DATA)
            
            # Save as Pickle
            pkl_path = os.path.join(RESULTS_DIR, f"{index_name}_data.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Saved {index_name} data to {pkl_path}")
            
            # Save as CSV for inspection
            csv_path = os.path.join(RESULTS_DIR, f"{index_name}_data.csv")
            df.to_csv(csv_path)
            logger.info(f"Saved {index_name} data to {csv_path}")
            
            # Verify data range
            start_date = df.index.min()
            end_date = df.index.max()
            logger.info(f"{index_name} Data Range: {start_date.date()} to {end_date.date()}")
            logger.info(f"Total rows: {len(df)}")
            
        except Exception as e:
            logger.error(f"Failed to fetch/save data for {index_name}: {e}")

if __name__ == "__main__":
    main()
