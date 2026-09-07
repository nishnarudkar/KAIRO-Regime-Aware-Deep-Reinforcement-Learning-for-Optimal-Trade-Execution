"""
Feature engineering and dataset preparation for Market Regime Detection.

Strict Causal Protocol:
- All features at timestamp t are computed using strictly data at or before t.
- Scalers are fitted ONLY on the chronological training split and applied out-of-sample.
- Chronological splitting preserves time-series order without future leakage.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


DEFAULT_REGIME_FEATURES = [
    "log_return",
    "realized_vol_15m",
    "parkinson_vol",
    "relative_volume_15m",
    "hl_spread_proxy",
]


class RegimeFeatureEngine:
    """
    Computes causal features tailored for latent market regime identification.
    
    Features capture:
    1. Volatility dynamics (realized rolling volatility, Parkinson high-low estimator)
    2. Return drift & magnitude (log returns)
    3. Liquidity & market activity (relative volume, Amihud illiquidity proxy)
    4. Spread / friction (high-low spread proxy)
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or DEFAULT_REGIME_FEATURES

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute causal regime features from raw OHLCV data.
        
        Args:
            df: DataFrame containing ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with computed causal features and dropped initial warm-up NaNs.
        """
        if df.empty:
            return df.copy()

        feat = df.copy()
        if "timestamp" in feat.columns and not pd.api.types.is_datetime64_any_dtype(feat["timestamp"]):
            feat["timestamp"] = pd.to_datetime(feat["timestamp"])

        # 1. Log Returns: r_t = ln(P_t / P_{t-1})
        if "log_return" not in feat.columns:
            feat["log_return"] = np.log(feat["close"] / feat["close"].shift(1))

        # 2. Rolling Realized Volatility (5m, 15m, 60m) - strictly backward-looking
        if "realized_vol_5m" not in feat.columns:
            feat["realized_vol_5m"] = feat["log_return"].rolling(window=5, min_periods=2).std()
        if "realized_vol_15m" not in feat.columns:
            feat["realized_vol_15m"] = feat["log_return"].rolling(window=15, min_periods=3).std()
        if "realized_vol_60m" not in feat.columns:
            feat["realized_vol_60m"] = feat["log_return"].rolling(window=60, min_periods=5).std()

        # 3. Parkinson High-Low Volatility: sigma_P = sqrt( (ln(H/L))^2 / (4 * ln(2)) )
        if "parkinson_vol" not in feat.columns:
            log_hl = np.log(feat["high"] / feat["low"].replace(0, np.nan))
            feat["parkinson_vol"] = np.sqrt((log_hl ** 2) / (4 * np.log(2)))

        # 4. Volume Dynamics: relative volume against backward rolling 15-period mean
        if "volume_ma_15m" not in feat.columns:
            feat["volume_ma_15m"] = feat["volume"].rolling(window=15, min_periods=1).mean()
        if "relative_volume_15m" not in feat.columns:
            feat["relative_volume_15m"] = feat["volume"] / (feat["volume_ma_15m"] + 1e-8)

        # 5. High-Low Spread Proxy: (High - Low) / Close
        if "hl_spread_proxy" not in feat.columns:
            feat["hl_spread_proxy"] = (feat["high"] - feat["low"]) / feat["close"]

        # 6. Amihud Illiquidity Proxy: |r_t| / (Volume_t * Close_t) * 1e6
        if "amihud_illiquidity" not in feat.columns:
            dollar_vol = feat["volume"] * feat["close"]
            feat["amihud_illiquidity"] = (feat["log_return"].abs() / (dollar_vol + 1.0)) * 1e6

        return feat

    def extract_feature_matrix(
        self,
        df: pd.DataFrame,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract the specified feature subset as a clean DataFrame and NumPy matrix.
        
        Args:
            df: Feature-engineered DataFrame.
            drop_na: Whether to drop rows with NaN in required feature columns.
            
        Returns:
            Tuple of (cleaned_df, feature_matrix).
        """
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        clean_df = df.copy()
        if drop_na:
            clean_df = clean_df.dropna(subset=self.feature_columns).reset_index(drop=True)

        X = clean_df[self.feature_columns].to_numpy(dtype=np.float64)
        return clean_df, X


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data strictly chronologically to prevent temporal lookahead.
    
    Args:
        df: DataFrame sorted by timestamp.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for out-of-sample testing.
        
    Returns:
        (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


class RegimeFeatureScaler:
    """
    Feature scaler that strictly fits on the training set and transforms new data
    without leaking future mean/variance statistics.
    """

    def __init__(self, method: str = "robust"):
        """
        Args:
            method: 'robust' (RobustScaler - median/IQR, outlier resistant)
                    or 'standard' (StandardScaler - mean/std)
        """
        self.method = method
        if method == "robust":
            self.scaler = RobustScaler()
        elif method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "RegimeFeatureScaler":
        """Fit scaler parameters on training data matrix."""
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned scaling to data matrix."""
        if not self.is_fitted:
            raise RuntimeError("RegimeFeatureScaler must be fitted before transforming data.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on training data and transform in one step."""
        return self.fit(X).transform(X)
