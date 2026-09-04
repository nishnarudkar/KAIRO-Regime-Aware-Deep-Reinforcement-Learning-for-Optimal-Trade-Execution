"""
Data storage manager for reading and writing raw/processed datasets in Parquet and CSV.
"""

from pathlib import Path
import pandas as pd

class DataStorage:
    """Manages file storage for raw and processed datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_bars(self, df: pd.DataFrame, symbol: str, filename_suffix: str = "") -> Path:
        """Save raw market bars to Parquet format."""
        filename = f"{symbol.lower()}_raw_{filename_suffix}.parquet" if filename_suffix else f"{symbol.lower()}_raw.parquet"
        filepath = self.raw_dir / filename
        df.to_parquet(filepath, index=False)
        return filepath

    def save_processed_features(self, df: pd.DataFrame, symbol: str, filename_suffix: str = "") -> Path:
        """Save processed market features to Parquet format."""
        filename = f"{symbol.lower()}_processed_{filename_suffix}.parquet" if filename_suffix else f"{symbol.lower()}_processed.parquet"
        filepath = self.processed_dir / filename
        df.to_parquet(filepath, index=False)
        return filepath

    def load_raw_bars(self, symbol: str, filename_suffix: str = "") -> pd.DataFrame:
        """Load raw market bars from Parquet format."""
        filename = f"{symbol.lower()}_raw_{filename_suffix}.parquet" if filename_suffix else f"{symbol.lower()}_raw.parquet"
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        return pd.read_parquet(filepath)

    def load_processed_features(self, symbol: str, filename_suffix: str = "") -> pd.DataFrame:
        """Load processed features from Parquet format."""
        filename = f"{symbol.lower()}_processed_{filename_suffix}.parquet" if filename_suffix else f"{symbol.lower()}_processed.parquet"
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed feature file not found: {filepath}")
        return pd.read_parquet(filepath)
