"""
Comprehensive unit tests for historical market data pipeline, cleaning, validation,
and lookahead leakage detection.
"""

from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.data.cleaning import DataCleaner
from src.data.validation import DataValidator
from src.features.pipeline import CausalFeatureEngine

def create_synthetic_test_bars() -> pd.DataFrame:
    """Helper fixture creating uncleaned synthetic bar data."""
    dates = pd.date_range(start="2026-01-01 09:30", periods=10, freq="1min", tz="UTC")
    # Intentional duplicate and out-of-order date
    raw_data = {
        "timestamp": list(dates) + [dates[3]],  # Duplicate timestamp
        "open": [150.0 + i for i in range(10)] + [153.0],
        "high": [151.0 + i for i in range(10)] + [154.0],
        "low": [149.0 + i for i in range(10)] + [152.0],
        "close": [150.5 + i for i in range(10)] + [153.5],
        "volume": [1000 + i * 100 for i in range(10)] + [1200],
    }
    df = pd.DataFrame(raw_data)
    # Shuffle order to test sorting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def test_chronological_sorting_and_deduplication():
    """Verify DataCleaner sorts timestamps and removes duplicates."""
    df_raw = create_synthetic_test_bars()
    cleaner = DataCleaner()
    df_clean = cleaner.clean_bars(df_raw, freq="1min")

    validator = DataValidator()
    assert validator.validate_chronological_order(df_clean)
    assert validator.validate_no_duplicates(df_clean)
    assert len(df_clean) == 10

def test_missing_data_grid_alignment():
    """Verify DataCleaner handles missing gaps with proper ffill/zero fill."""
    dates = pd.date_range(start="2026-01-01 09:30", periods=5, freq="2min", tz="UTC")  # 2min gap
    df_gap = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [500] * 5,
    })

    cleaner = DataCleaner()
    df_clean = cleaner.clean_bars(df_gap, freq="1min")

    # Regular 1-min grid over 5 2-min bars spans 9 minutes total (9 bars)
    assert len(df_clean) == 9
    assert df_clean["close"].isna().sum() == 0
    # Filled volume for gap bars should be 0
    assert (df_clean["volume"] == 0).sum() == 4

def test_strict_anti_lookahead_causality():
    """
    CRITICAL TEST: Verify feature calculation at time t depends ONLY on data <= t.
    Mutating future prices (t+1 .. T) must NOT change feature values at timestamp t.
    """
    dates = pd.date_range(start="2026-01-01 09:30", periods=30, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(150, 1, 30),
        "high": np.random.normal(151, 1, 30),
        "low": np.random.normal(149, 1, 30),
        "close": np.random.normal(150, 1, 30),
        "volume": np.random.randint(1000, 5000, 30),
    })

    validator = DataValidator()
    feature_cols = ["log_return", "realized_vol_5m", "realized_vol_15m", "parkinson_vol", "relative_volume_15m"]
    is_causal = validator.check_lookahead_leakage(df, feature_cols)

    assert is_causal, "Lookahead leakage detected in causal feature engine!"
