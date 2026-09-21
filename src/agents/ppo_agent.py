"""
PPO Agent — Stage 9.

WHY PPO FOR THIS RESEARCH:
  DQN is an off-policy value-based algorithm. PPO is an on-policy policy-gradient
  algorithm. If the performance pattern observed with DQN (e.g., regime-aware
  improvement) also appears with PPO, the finding is algorithmic-class-agnostic
  and therefore more credible.

  Research question addressed:
    RQ4: Is the regime-aware performance improvement specific to DQN, or does
         it generalise to a second RL algorithm class (PPO)?

  If Regime-Aware PPO (Model D) also outperforms plain PPO (Model A-PPO),
  the evidence for RQ2 is strengthened independently.

DESIGN CONSTRAINTS (from Project.md Stage 9):
  - Identical experimental conditions to DQN experiments (same env, reward,
    data split, evaluation metrics, baselines)
  - No redesign of the existing pipeline
  - PPO + Regime-Aware PPO evaluated; SAC and other algorithms NOT added
    without a clear research justification

IMPLEMENTATION:
  Wraps Stable-Baselines3 PPO (on-policy, continuous or discrete action support).
  Action space remains Discrete(4) — identical to DQN.
  Uses MlpPolicy with the same net_arch [128, 128] as DQN for a fair comparison.

HYPERPARAMETER CHOICES (documented explicitly):
  n_steps=512        — rollout buffer length (on-policy requires collecting
                       full rollouts before updating)
  batch_size=64      — mini-batch size for the policy update
  n_epochs=10        — number of epochs per rollout
  gamma=0.99         — discount (same as DQN)
  gae_lambda=0.95    — GAE advantage estimation lambda
  clip_range=0.2     — PPO clipping coefficient (standard from Schulman 2017)
  learning_rate=3e-4 — Adam LR (PPO typically uses higher LR than DQN)
  net_arch=[128,128] — MLP architecture (identical to DQN for fair comparison)

All hyperparameters are documented here and in config/ppo.yaml.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from src.agents import BaseAgentWrapper


class PPOAgent(BaseAgentWrapper):
    """
    Stable-Baselines3 PPO agent for discrete trade-execution actions.

    Compatible with both TradeExecutionEnv (7-dim obs, Stage 6/9)
    and RegimeAwareTradeExecutionEnv (12-dim obs, Stage 7/9).

    Args:
        env: Gymnasium-compatible environment.
        seed: Integer random seed for full determinism.
        learning_rate: Adam learning rate.
        n_steps: Steps to collect per rollout before update.
        batch_size: Mini-batch size for policy gradient updates.
        n_epochs: Gradient epochs per rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping coefficient.
        policy_kwargs: Extra kwargs for the MLP policy network.
        verbose: SB3 verbosity (0=quiet, 1=info).
    """

    def __init__(
        self,
        env,
        seed: int = 42,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        from stable_baselines3 import PPO

        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [128, 128]}

        self.env = env
        self.seed = seed

        self._model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=verbose,
        )

    # ── BaseAgentWrapper interface ─────────────────────────────────────────────

    def train(self, total_timesteps: int, callback=None, progress_bar: bool = False) -> None:
        """
        Train the PPO policy.

        Args:
            total_timesteps: Total environment interaction steps.
            callback: Optional SB3 callback.
        """
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=progress_bar,
        )

    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Return the greedy (or stochastic) action for a given state.

        Args:
            state: Observation array compatible with the env's observation space.
            deterministic: If True, use the mode of the policy distribution.

        Returns:
            Integer action index in {0, 1, 2, 3}.
        """
        action, _ = self._model.predict(state, deterministic=deterministic)
        return int(action)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the trained model to disk (SB3 adds .zip automatically)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env) -> "PPOAgent":
        """Load a previously saved PPO model from disk."""
        from stable_baselines3 import PPO

        instance = cls.__new__(cls)
        instance.env = env
        instance._model = PPO.load(path, env=env)
        return instance

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def model(self):
        """Expose underlying SB3 PPO model for callbacks / inspection."""
        return self._model
