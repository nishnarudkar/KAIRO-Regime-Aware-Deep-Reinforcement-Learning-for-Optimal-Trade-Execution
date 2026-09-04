"""
Validation module for checking data integrity, chronological order, non-negative values,
and strict anti-lookahead leakage causality.
"""

import pandas as pd
import numpy as np

class DataValidator:
    """Validates data integrity and anti-lookahead rules."""

    @staticmethod
    def validate_chronological_order(df: pd.DataFrame) -> bool:
        """Assert timestamps are strictly monotonically increasing."""
        if df.empty or "timestamp" not in df.columns:
            return True
        return df["timestamp"].is_monotonic_increasing

    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame) -> bool:
        """Assert no duplicate timestamps exist."""
        if df.empty or "timestamp" not in df.columns:
            return True
        return not df["timestamp"].duplicated().any()

    @staticmethod
    def validate_non_negative_prices(df: pd.DataFrame) -> bool:
        """Assert price and volume columns contain non-negative numbers."""
        price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        for col in price_cols:
            if (df[col] < 0).any():
                return False
        return True

    @staticmethod
    def check_lookahead_leakage(df: pd.DataFrame, feature_cols: list[str]) -> bool:
        """
        Verify causal anti-lookahead requirement:
        Mutating future prices (t+1 .. T) MUST NOT change feature value at timestamp t.
        Returns True if strictly causal (no leakage detected), False otherwise.
        """
        if df.empty or len(df) < 5:
            return True

        from src.features.pipeline import CausalFeatureEngine

        df_original = df.copy()
        engine = CausalFeatureEngine()
        df_feat_orig = engine.compute_features(df_original)

        # Mutate the last row's close price heavily
        df_mutated = df_original.copy()
        last_idx = df_mutated.index[-1]
        df_mutated.loc[last_idx, "close"] = df_mutated.loc[last_idx, "close"] * 10.0
        df_mutated.loc[last_idx, "high"] = df_mutated.loc[last_idx, "high"] * 10.0

        df_feat_mutated = engine.compute_features(df_mutated)

        # Features at timestamps strictly BEFORE last_idx MUST be identical
        t_before = df_feat_orig.index[:-1]
        for col in feature_cols:
            if col in df_feat_orig.columns:
                orig_vals = df_feat_orig.loc[t_before, col].dropna()
                mut_vals = df_feat_mutated.loc[t_before, col].dropna()
                if not np.allclose(orig_vals.values, mut_vals.values, rtol=1e-5, atol=1e-5):
                    return False  # Leakage detected!

        return True
