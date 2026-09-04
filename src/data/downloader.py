"""
Downloader module for fetching historical intraday market data (bars/OHLCV) from Alpaca API.
"""

from datetime import datetime
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.data.alpaca_client import get_alpaca_data_client

class AlpacaDataDownloader:
    """Downloader class for fetching stock bars from Alpaca API."""

    def __init__(self, client=None):
        self.client = client or get_alpaca_data_client()

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
        feed: str = "iex"
    ) -> pd.DataFrame:
        """
        Fetch historical stock bars for a target symbol.
        Default feed is 'iex' for free-tier Alpaca account compatibility.
        """
        from alpaca.data.enums import DataFeed

        feed_enum = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=feed_enum
        )

        bars = self.client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            return pd.DataFrame()

        # If MultiIndex (symbol, timestamp), reset index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Rename timestamp column if needed
        if "timestamp" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})

        # Ensure UTC timezone awareness and proper sorting
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df
