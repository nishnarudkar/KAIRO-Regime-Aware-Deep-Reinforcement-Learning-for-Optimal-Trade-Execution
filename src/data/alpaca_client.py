"""
Alpaca Historical Data Client wrapper.
Loads credentials safely from environment variables or .env file.
"""

import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv()

def get_alpaca_data_client() -> StockHistoricalDataClient:
    """
    Initialize and return an Alpaca StockHistoricalDataClient instance
    using environment variables ALPACA_API_KEY_ID and ALPACA_SECRET_KEY.
    """
    api_key = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials missing. Please set ALPACA_API_KEY_ID "
            "and ALPACA_SECRET_KEY in environment or .env file."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
