"""
Causal Feature Engineering Engine.
Strictly ensures that at timestamp t, every feature is calculated using ONLY data at or before t.
"""

import numpy as np
import pandas as pd
from src.features import BaseFeatureCalculator

class CausalFeatureEngine(BaseFeatureCalculator):
    """Computes technical, microstructural, and volatility features without lookahead leakage."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input DataFrame must contain: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Output DataFrame contains input columns plus calculated causal features.
        """
        if df.empty:
            return df

        feat_df = df.copy()

        # 1. Log Returns: r_t = ln(P_t / P_{t-1})
        feat_df["log_return"] = np.log(feat_df["close"] / feat_df["close"].shift(1))

        # 2. Rolling Realized Volatility (5m, 15m, 60m windows)
        # Using backward-looking rolling windows on returns
        feat_df["realized_vol_5m"] = feat_df["log_return"].rolling(window=5, min_periods=2).std()
        feat_df["realized_vol_15m"] = feat_df["log_return"].rolling(window=15, min_periods=3).std()
        feat_df["realized_vol_60m"] = feat_df["log_return"].rolling(window=60, min_periods=5).std()

        # 3. Parkinson High-Low Volatility Estimator (Causal, uses current high & low)
        # sigma^2_P = (ln(H_t / L_t))^2 / (4 * ln(2))
        log_hl = np.log(feat_df["high"] / feat_df["low"].replace(0, np.nan))
        feat_df["parkinson_vol"] = np.sqrt((log_hl ** 2) / (4 * np.log(2)))

        # 4. Volume Features (Rolling relative volume ratio V_t / E[V])
        feat_df["volume_ma_15m"] = feat_df["volume"].rolling(window=15, min_periods=1).mean()
        feat_df["relative_volume_15m"] = feat_df["volume"] / (feat_df["volume_ma_15m"] + 1e-8)

        # 5. High-Low Spread Proxy: (High - Low) / Close
        feat_df["hl_spread_proxy"] = (feat_df["high"] - feat_df["low"]) / feat_df["close"]

        # 6. Amihud Illiquidity Proxy: |Return_t| / (Volume_t * Close_t)
        dollar_volume = feat_df["volume"] * feat_df["close"]
        feat_df["amihud_illiquidity"] = (feat_df["log_return"].abs() / (dollar_volume + 1.0)) * 1e6

        return feat_df
