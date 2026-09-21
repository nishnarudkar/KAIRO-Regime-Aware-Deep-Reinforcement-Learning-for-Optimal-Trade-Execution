"""
src/evaluation/ablation.py — Ablation Study: Shuffled-Regime Control (Model C)

Model C is the critical scientific control for RQ2.

PURPOSE:
  If Regime-Aware DQN (Model B) outperforms plain DQN (Model A), we must
  determine WHETHER the improvement comes from:
    (a) Genuinely useful regime information, OR
    (b) Simply having an extra feature dimension in the state

Model C resolves this by replacing the causally-inferred regime label with
RANDOMLY SHUFFLED regime labels — same feature vector size as Model B, but
the regime signal is now informationally worthless noise.

RESULT INTERPRETATION:
  - If Model B >> Model C: regime information is causally useful
  - If Model B ≈ Model C: improvement is from extra dimensionality, not regime

METHODOLOGY:
  - Shuffle the regime ID and regime probability columns independently at each
    timestep using a fixed seed (not correlated with price data)
  - State dimension remains 12 (identical to Model B)
  - All other conditions identical: same data, same hyperparams, same seed

HARD RULE COMPLIANCE:
  This control does NOT modify any real result. It trains a fresh agent on
  the shuffled-regime environment and reports the result as-is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.environment.env import TradeExecutionEnv
from src.execution import BaseImpactModel, default_impact_model
from src.environment.rewards import ModularExecutionReward, default_reward_calculator

logger = logging.getLogger(__name__)


class ShuffledRegimeEnv(gym.Env):
    """
    Model C: DQN with shuffled (uninformative) regime labels.

    The observation vector has the same 12-dim shape as RegimeAwareTradeExecutionEnv,
    but indices [7:12] contain randomly shuffled, non-causal regime noise.

    This env is ONLY used for the ablation control experiment.
    It must NOT be used for any real execution task.
    """

    metadata = {"render_modes": []}
    N_OBS = 12   # same size as Model B

    def __init__(
        self,
        n_regimes: int = 4,
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
        shuffle_seed: int = 999,    # fixed seed for shuffled noise (NOT tied to price)
        random_start: bool = False,
        start_range=None,
        history_bars: int = 60,
        order_participation=None,
    ):
        super().__init__()

        self.n_regimes = n_regimes
        self.shuffle_rng = np.random.default_rng(shuffle_seed)

        self.action_space = spaces.Discrete(len(action_fractions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32
        )

        self._base_env = TradeExecutionEnv(
            market_data=market_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
            action_fractions=action_fractions,
            impact_model=impact_model or default_impact_model(),
            reward_calculator=reward_calculator or default_reward_calculator(),
            max_participation_rate=max_participation_rate,
            per_share_fee=per_share_fee,
            default_spread_bps=default_spread_bps,
            random_start=random_start,
            start_range=start_range,
            history_bars=history_bars,
            order_participation=order_participation,
        )

    def _random_regime_features(self) -> np.ndarray:
        """
        Generate random (uninformative) regime features.

        - regime_id: random integer in [0, n_regimes)
        - regime_probs: random Dirichlet draw (sums to 1, non-negative)
          but NOT correlated with market state
        """
        regime_id = float(self.shuffle_rng.integers(0, self.n_regimes))
        # Dirichlet with alpha=1 gives uniform random simplex point
        probs = self.shuffle_rng.dirichlet(np.ones(self.n_regimes)).astype(np.float32)
        return np.array([regime_id] + probs.tolist(), dtype=np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_obs, info = self._base_env.reset(seed=seed, options=options)
        noise = self._random_regime_features()
        obs = np.concatenate([base_obs, noise], dtype=np.float32)
        info["shuffled_regime"] = True
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        base_obs, reward, terminated, truncated, info = self._base_env.step(action)
        noise = self._random_regime_features()
        obs = np.concatenate([base_obs, noise], dtype=np.float32)
        info["shuffled_regime"] = True
        return obs, reward, terminated, truncated, info

    @property
    def simulator(self):
        return self._base_env.simulator


def run_ablation_control(
    market_data: pd.DataFrame,
    train_timesteps: int = 50_000,
    seed: int = 42,
    verbose: int = 0,
) -> Any:
    """
    Train Model C (shuffled-regime DQN) and return its AgentEvalResult.

    Args:
        market_data: Training-split data only.
        train_timesteps: Training duration.
        seed: Random seed for DQN training.
        verbose: SB3 verbosity.

    Returns:
        AgentEvalResult with agent_name = "DQN (shuffled regime)"
    """
    from src.agents.dqn_agent import DQNAgent
    from src.agents.evaluator import evaluate_agent

    logger.info("Training Model C: DQN with shuffled regime labels …")

    env = ShuffledRegimeEnv(
        market_data=market_data,
        target_inventory=100_000.0,
        side="BUY",
        horizon_steps=30,
        shuffle_seed=seed + 1000,  # different from DQN seed but fixed
    )

    agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=200)
    agent.train(total_timesteps=train_timesteps)

    result = evaluate_agent(
        agent=agent,
        market_data=market_data,
        target_inventory=100_000.0,
        side="BUY",
        horizon_steps=30,
        seed=seed,
        agent_name="DQN (shuffled regime)",
    )

    logger.info(f"  Model C done. IS={result.implementation_shortfall_bps:.2f} bps")
    return result
