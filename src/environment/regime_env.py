"""
RegimeAwareTradeExecutionEnv — Stage 7

Extends TradeExecutionEnv by injecting causally-inferred HMM regime information
into the observation vector.

CRITICAL LEAKAGE PREVENTION:
  At every timestep t, the regime label is obtained from CausalRegimeInference
  which runs the HMM Forward Algorithm:
      P(S_t | X_{1:t}) = normalize( b(X_t) * (P(S_{t-1} | X_{1:t-1}) @ A) )
  This guarantees that S_t depends ONLY on observations up to and including t.
  Future observations X_{t+1:T} have ZERO influence on S_t.

Observation Space: Box(12,)
  Indices 0–6: Identical to TradeExecutionEnv (market + order features)
  Index  7:    Regime ID (integer cast to float32, range 0–3)
  Indices 8–11: Regime posterior probabilities [P(S=0|X_{1:t}) .. P(S=3|X_{1:t})]

The environment ID "regime_aware" is used to distinguish it from the base env
in MLflow run configs and experiment comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.environment.env import TradeExecutionEnv
from src.execution import BaseImpactModel, AlmgrenChrissImpactModel
from src.environment.rewards import ModularExecutionReward


class RegimeAwareTradeExecutionEnv(gym.Env):
    """
    Gymnasium Trade Execution MDP with causal HMM regime information in the state.

    Model B in the Stage 7 A/B experiment.
    Keep ALL other conditions identical to TradeExecutionEnv (Model A):
      - Same action space
      - Same reward function and coefficients
      - Same market data
      - Same horizon, target inventory, impact model

    Only difference: observation vector is extended with regime features.

    Args:
        hmm_model: Fitted MarketHMM instance (trained on train split ONLY).
        regime_feature_data: DataFrame used to step through regime features.
                             Must be aligned row-by-row with market_data.
                             Columns: DEFAULT_REGIME_FEATURES (log_return,
                             realized_vol_15m, parkinson_vol, relative_volume_15m,
                             hl_spread_proxy).
        market_data: Same market data as used in TradeExecutionEnv.
        target_inventory: Total shares to execute.
        side: 'BUY' or 'SELL'.
        horizon_steps: Episode length in steps.
        action_fractions: Discrete fraction choices.
        impact_model: Shared market impact model.
        reward_calculator: Modular reward function.
        max_participation_rate: Simulator constraint.
        per_share_fee: Commission per share.
        default_spread_bps: Fallback spread.
    """

    metadata = {"render_modes": []}

    N_BASE_OBS = 7   # from TradeExecutionEnv
    N_REGIME_OBS = 5  # regime_id + 4 probabilities
    N_OBS = N_BASE_OBS + N_REGIME_OBS  # 12 total

    def __init__(
        self,
        hmm_model,
        regime_feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        target_inventory: float = 100_000.0,
        side: str = "BUY",
        horizon_steps: int = 30,
        action_fractions: List[float] = [0.0, 0.10, 0.25, 0.50],
        impact_model: Optional[BaseImpactModel] = None,
        reward_calculator: Optional[ModularExecutionReward] = None,
        max_participation_rate: float = 0.15,
        per_share_fee: float = 0.0005,
        default_spread_bps: float = 2.0,
    ):
        super().__init__()

        self.hmm_model = hmm_model
        self.regime_feature_data = regime_feature_data
        self.market_data = market_data
        self.target_inventory = float(target_inventory)
        self.side = side.upper()
        self.horizon_steps = int(horizon_steps)
        self.action_fractions = action_fractions
        self.num_actions = len(action_fractions)

        # Action space: identical to base env
        self.action_space = spaces.Discrete(self.num_actions)

        # Extended observation space: 12-dimensional
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_OBS,),
            dtype=np.float32,
        )

        # Shared sub-systems
        self.impact_model = impact_model or AlmgrenChrissImpactModel(eta=0.05, gamma=0.01)
        self.reward_calculator = reward_calculator or ModularExecutionReward()

        # Delegate market replay / execution to base env
        self._base_env = TradeExecutionEnv(
            market_data=market_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
            action_fractions=action_fractions,
            impact_model=self.impact_model,
            reward_calculator=self.reward_calculator,
            max_participation_rate=max_participation_rate,
            per_share_fee=per_share_fee,
            default_spread_bps=default_spread_bps,
        )

        # Causal inference engine — initialised lazily after HMM check
        self._causal_inference = None
        self._regime_feature_rows: Optional[np.ndarray] = None
        self._current_step: int = 0
        self._current_regime_id: int = 0
        self._current_regime_probs: np.ndarray = np.ones(4) / 4  # uniform prior

        self._init_causal_inference()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_causal_inference(self) -> None:
        """Validate HMM and build CausalRegimeInference engine."""
        from src.regimes.inference import CausalRegimeInference

        if not self.hmm_model.is_fitted:
            raise RuntimeError(
                "hmm_model must be fitted before creating RegimeAwareTradeExecutionEnv. "
                "Train the HMM on the training split first."
            )
        self._causal_inference = CausalRegimeInference(self.hmm_model)

    def _prepare_regime_features(self, df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Extract and cache regime feature matrix from regime_feature_data.
        Returns numpy array of shape (N, D) or None if no regime data provided.
        """
        if df is None:
            return None

        from src.regimes.features import RegimeFeatureEngine

        engine = RegimeFeatureEngine(feature_columns=self.hmm_model.feature_names)
        try:
            _, X = engine.extract_feature_matrix(df, drop_na=False)
            return X
        except Exception:
            return None

    # ── Gymnasium Interface ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset to a new episode.

        Resets both the base environment and the causal regime inference state
        so the forward filter starts fresh (no carryover from previous episode).
        """
        # Reset causal inference — CRITICAL: must start forward filter from scratch
        self._causal_inference.reset_online_state()
        self._current_step = 0
        self._current_regime_id = 0
        self._current_regime_probs = np.ones(self.hmm_model.n_regimes) / self.hmm_model.n_regimes

        # Prepare regime feature rows for this episode
        regime_df = None
        if options and "regime_feature_data" in options:
            regime_df = options["regime_feature_data"]
        elif self.regime_feature_data is not None:
            regime_df = self.regime_feature_data

        self._regime_feature_rows = self._prepare_regime_features(regime_df)

        # Reset base env
        base_obs, base_info = self._base_env.reset(seed=seed, options=options)

        # Step regime engine for first timestep
        self._step_regime()

        obs = self._extend_observation(base_obs)
        info = self._extend_info(base_info)

        return obs, info

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform one environment step.
        Regime is updated causally before constructing the next observation.
        """
        base_obs, reward, terminated, truncated, info = self._base_env.step(action)

        self._current_step += 1

        # Update regime belief for the new timestep (causal: uses current row only)
        if not (terminated or truncated):
            self._step_regime()

        obs = self._extend_observation(base_obs)
        info = self._extend_info(info)

        return obs, reward, terminated, truncated, info

    # ── Regime Stepping ────────────────────────────────────────────────────────

    def _step_regime(self) -> None:
        """
        Advance HMM forward filter by one step.

        If regime_feature_rows is available, uses the actual feature vector at
        the current timestep. Otherwise falls back to a zero vector (uniform belief).

        LEAKAGE GUARANTEE: Only self._regime_feature_rows[self._current_step]
        is passed — never any future row.
        """
        if self._regime_feature_rows is not None:
            idx = min(self._current_step, len(self._regime_feature_rows) - 1)
            x_t = self._regime_feature_rows[idx]

            # Replace any NaN (warmup period) with zeros — safe default
            x_t = np.where(np.isnan(x_t), 0.0, x_t)
        else:
            # Fallback: zero features → uniform regime posterior
            n_features = len(self.hmm_model.feature_names)
            x_t = np.zeros(n_features, dtype=np.float64)

        regime_id, probs = self._causal_inference.step_online(x_t)
        self._current_regime_id = int(regime_id)
        self._current_regime_probs = probs

    # ── Observation Assembly ───────────────────────────────────────────────────

    def _extend_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """
        Extend 7-dim base observation with regime features.

        Extended vector layout:
          [0:7]  base market + order features (identical to TradeExecutionEnv)
          [7]    regime_id as float (0.0, 1.0, 2.0, or 3.0)
          [8:12] regime posterior probabilities [P0, P1, P2, P3]
        """
        n_regimes = self.hmm_model.n_regimes
        probs = np.array(self._current_regime_probs, dtype=np.float32)

        # Pad/trim in case n_regimes != 4 (should not happen but defensive)
        if len(probs) < 4:
            probs = np.concatenate([probs, np.zeros(4 - len(probs), dtype=np.float32)])
        elif len(probs) > 4:
            probs = probs[:4]

        regime_features = np.array(
            [float(self._current_regime_id)] + probs.tolist(),
            dtype=np.float32,
        )

        return np.concatenate([base_obs, regime_features], dtype=np.float32)

    def _extend_info(self, base_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add regime metadata to info dict."""
        regime_label = self.hmm_model.regime_labels.get(
            self._current_regime_id, f"Regime_{self._current_regime_id}"
        )
        base_info.update({
            "regime_id": self._current_regime_id,
            "regime_label": regime_label,
            "regime_probs": self._current_regime_probs.tolist(),
        })
        return base_info

    # ── Convenience Accessors ──────────────────────────────────────────────────

    @property
    def simulator(self):
        """Expose base env's simulator for metric collection."""
        return self._base_env.simulator

    @property
    def current_regime_id(self) -> int:
        return self._current_regime_id

    @property
    def current_regime_label(self) -> str:
        return self.hmm_model.regime_labels.get(
            self._current_regime_id, f"Regime_{self._current_regime_id}"
        )
