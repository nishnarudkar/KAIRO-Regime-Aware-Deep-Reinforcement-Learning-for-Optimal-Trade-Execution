"""
Data cleaning module for sorting, deduplicating, missing value handling, and time grid alignment.
"""

import pandas as pd

class DataCleaner:
    """Cleans and standardizes raw market bar data DataFrames."""

    @staticmethod
    def clean_bars(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """
        Clean market bar data:
        1. Sort chronologically by timestamp.
        2. Remove duplicate timestamps (keeping last).
        3. Fill missing time steps on regular grid using forward-fill for prices, zero-fill for volume.
        """
        if df.empty:
            return df

        cleaned_df = df.copy()

        # Ensure timestamp is datetime & UTC
        cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"], utc=True)

        # Sort chronologically
        cleaned_df = cleaned_df.sort_values("timestamp")

        # Deduplicate timestamps
        cleaned_df = cleaned_df.drop_duplicates(subset=["timestamp"], keep="last")

        # Set timestamp index for grid alignment
        cleaned_df = cleaned_df.set_index("timestamp")

        # Create complete regular time grid between start and end timestamps
        full_idx = pd.date_range(
            start=cleaned_df.index.min(),
            end=cleaned_df.index.max(),
            freq=freq,
            tz="UTC",
            name="timestamp"
        )

        # Reindex to regular time grid
        cleaned_df = cleaned_df.reindex(full_idx)

        # Forward fill prices, zero fill volume
        price_cols = [c for c in ["open", "high", "low", "close", "vwap"] if c in cleaned_df.columns]
        cleaned_df[price_cols] = cleaned_df[price_cols].ffill().bfill()

        if "volume" in cleaned_df.columns:
            cleaned_df["volume"] = cleaned_df["volume"].fillna(0)

        if "trade_count" in cleaned_df.columns:
            cleaned_df["trade_count"] = cleaned_df["trade_count"].fillna(0)

        # Reset index to return timestamp column
        cleaned_df = cleaned_df.reset_index()

        return cleaned_df
