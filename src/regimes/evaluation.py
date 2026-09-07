"""
Evaluation and Diagnostic Metrics for Market Regime Detection.

Provides:
- Empirical regime statistics (mean returns, volatility, volume, illiquidity per state)
- Transition probability matrix diagnostics and stationary distribution computation
- Regime duration and persistence metrics
- Out-of-sample log-likelihood and information criteria (AIC/BIC)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.regimes.hmm_model import MarketHMM


class RegimeEvaluator:
    """
    Computes diagnostic and descriptive evaluation metrics for market regime models.
    """

    def __init__(self, hmm_model: MarketHMM):
        """
        Args:
            hmm_model: Trained MarketHMM instance.
        """
        if not hmm_model.is_fitted:
            raise RuntimeError("MarketHMM must be fitted before running evaluation.")
        self.hmm = hmm_model

    def evaluate_dataset(
        self,
        df_with_regimes: pd.DataFrame,
        regime_col: str = "causal_regime"
    ) -> Dict[str, Any]:
        """
        Compute empirical feature statistics grouped by detected regime.
        
        Args:
            df_with_regimes: DataFrame containing features and inferred regime column.
            regime_col: Name of column containing discrete regime IDs.
            
        Returns:
            Dictionary containing regime breakdown and profile statistics.
        """
        if regime_col not in df_with_regimes.columns:
            raise ValueError(f"Column '{regime_col}' not found in DataFrame.")

        total_obs = len(df_with_regimes)
        stats_per_regime = {}

        for k in range(self.hmm.n_regimes):
            label = self.hmm.regime_labels.get(k, f"Regime_{k}")
            sub = df_with_regimes[df_with_regimes[regime_col] == k]
            count = len(sub)
            freq = count / total_obs if total_obs > 0 else 0.0

            prof: Dict[str, float] = {
                "label": label,
                "count": count,
                "frequency": freq,
            }

            if count > 0:
                if "log_return" in sub.columns:
                    prof["mean_return"] = float(sub["log_return"].mean())
                    prof["volatility"] = float(sub["log_return"].std())
                    prof["ann_volatility"] = float(sub["log_return"].std() * np.sqrt(252 * 390))
                if "parkinson_vol" in sub.columns:
                    prof["mean_parkinson_vol"] = float(sub["parkinson_vol"].mean())
                if "realized_vol_15m" in sub.columns:
                    prof["mean_realized_vol_15m"] = float(sub["realized_vol_15m"].mean())
                if "relative_volume_15m" in sub.columns:
                    prof["mean_relative_volume"] = float(sub["relative_volume_15m"].mean())
                if "hl_spread_proxy" in sub.columns:
                    prof["mean_spread_proxy"] = float(sub["hl_spread_proxy"].mean())
                if "amihud_illiquidity" in sub.columns:
                    prof["mean_amihud_illiquidity"] = float(sub["amihud_illiquidity"].mean())

            stats_per_regime[k] = prof

        # Transition matrix stats
        A = self.hmm.transition_matrix
        stationary = self.hmm.stationary_distribution
        durations = self.hmm.expected_durations

        return {
            "total_observations": total_obs,
            "regime_profiles": stats_per_regime,
            "transition_matrix": A.tolist(),
            "stationary_distribution": stationary.tolist(),
            "expected_durations_bars": durations,
        }

    def generate_summary_table(
        self,
        df_with_regimes: pd.DataFrame,
        regime_col: str = "causal_regime"
    ) -> pd.DataFrame:
        """
        Generate a clean summary DataFrame profiling each regime.
        """
        eval_dict = self.evaluate_dataset(df_with_regimes, regime_col=regime_col)
        profiles = eval_dict["regime_profiles"]

        rows = []
        for k, prof in profiles.items():
            row = {
                "Regime ID": k,
                "Regime Name": prof["label"],
                "Count": prof["count"],
                "Frequency (%)": f"{prof['frequency'] * 100:.2f}%",
                "Mean Return (bps)": f"{prof.get('mean_return', 0.0) * 1e4:.2f}",
                "Volatility (bps)": f"{prof.get('volatility', 0.0) * 1e4:.2f}",
                "Parkinson Vol": f"{prof.get('mean_parkinson_vol', 0.0):.6f}",
                "Rel Volume": f"{prof.get('mean_relative_volume', 0.0):.2f}",
                "Spread Proxy": f"{prof.get('mean_spread_proxy', 0.0):.6f}",
                "Exp Duration (bars)": f"{eval_dict['expected_durations_bars'].get(k, 0.0):.1f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)
