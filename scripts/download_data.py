"""
CLI script to download, clean, validate, extract features, and save historical market data from Alpaca.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.downloader import AlpacaDataDownloader
from src.data.cleaning import DataCleaner
from src.data.validation import DataValidator
from src.data.storage import DataStorage
from src.features.pipeline import CausalFeatureEngine

def main():
    parser = argparse.ArgumentParser(description="Download and process historical intraday bar data from Alpaca.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--days", type=int, default=5, help="Number of historical days to fetch (default: 5)")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    print(f"Downloading historical 1-minute bars for {args.symbol} from {start_date.date()} to {end_date.date()}...")

    # 1. Download raw data
    downloader = AlpacaDataDownloader()
    raw_df = downloader.fetch_bars(args.symbol, start_date=start_date, end_date=end_date)

    if raw_df.empty:
        print(f"No market data returned for symbol {args.symbol}.")
        return

    print(f"Downloaded {len(raw_df)} raw 1-minute bars.")

    # 2. Clean data
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_bars(raw_df, freq="1min")
    print(f"Cleaned and regularized data: {len(cleaned_df)} bars.")

    # 3. Validate data integrity
    validator = DataValidator()
    assert validator.validate_chronological_order(cleaned_df), "Validation Failed: Data is not in chronological order!"
    assert validator.validate_no_duplicates(cleaned_df), "Validation Failed: Duplicate timestamps found!"
    assert validator.validate_non_negative_prices(cleaned_df), "Validation Failed: Negative price values found!"
    print("Data integrity validation passed successfully.")

    # 4. Extract causal features
    feature_engine = CausalFeatureEngine()
    processed_df = feature_engine.compute_features(cleaned_df)
    print(f"Extracted {len(processed_df.columns)} columns/features.")

    # 5. Anti-lookahead leakage assertion check
    feature_cols = ["log_return", "realized_vol_5m", "realized_vol_15m", "parkinson_vol", "relative_volume_15m"]
    is_causal = validator.check_lookahead_leakage(cleaned_df, feature_cols)
    assert is_causal, "CRITICAL ERROR: Lookahead leakage detected in feature calculation!"
    print("Anti-lookahead leakage check PASSED (features are strictly causal).")

    # 6. Save raw and processed datasets to Parquet
    storage = DataStorage()
    date_suffix = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    raw_path = storage.save_raw_bars(raw_df, symbol=args.symbol, filename_suffix=date_suffix)
    processed_path = storage.save_processed_features(processed_df, symbol=args.symbol, filename_suffix=date_suffix)

    print(f"Raw data saved to: {raw_path}")
    print(f"Processed feature data saved to: {processed_path}")

if __name__ == "__main__":
    main()
