"""
DQN Agent — Stage 6 (no regime information).

Wraps Stable-Baselines3 DQN to provide:
  - Deterministic seeding
  - Configurable hyperparameters
  - Model checkpoint save / load
  - Predict interface compatible with BaseAgentWrapper
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.agents import BaseAgentWrapper


class DQNAgent(BaseAgentWrapper):
    """
    Stable-Baselines3 DQN agent for discrete trade-execution actions.

    The agent operates inside TradeExecutionEnv (Gymnasium).
    No regime information is passed — regime-awareness is Stage 7.

    Args:
        env: A Gymnasium-compatible TradeExecutionEnv instance.
        seed: Integer random seed for full determinism.
        learning_rate: Adam learning rate (default 1e-4 from DQN literature).
        buffer_size: Replay buffer capacity.
        learning_starts: Steps before first gradient update.
        batch_size: Mini-batch size per gradient update.
        tau: Soft-update coefficient for target network.
        gamma: Discount factor.
        train_freq: Update the model every `train_freq` steps.
        gradient_steps: Gradient steps per env step.
        target_update_interval: Steps between hard target-network copies.
        exploration_fraction: Fraction of training with epsilon decay.
        exploration_initial_eps: Starting epsilon.
        exploration_final_eps: Final epsilon.
        policy_kwargs: Extra kwargs forwarded to the MLP policy network.
        verbose: SB3 verbosity level (0=quiet, 1=info, 2=debug).
    """

    def __init__(
        self,
        env,
        seed: int = 42,
        learning_rate: float = 1e-4,
        buffer_size: int = 50_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1_000,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        from stable_baselines3 import DQN

        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [128, 128]}

        self.env = env
        self.seed = seed

        self._model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # BaseAgentWrapper interface
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int, callback=None, progress_bar: bool = False) -> None:
        """
        Train the DQN policy.

        Args:
            total_timesteps: Total environment steps to train for.
            callback: Optional SB3 callback (e.g. EvalCallback).
        """
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=progress_bar,
        )

    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Return the greedy (or epsilon-greedy) action for a given state.

        Args:
            state: Observation array of shape (7,).
            deterministic: If True, use greedy policy (no epsilon exploration).

        Returns:
            Integer action index in {0, 1, 2, 3}.
        """
        action, _ = self._model.predict(state, deterministic=deterministic)
        return int(action)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path (without extension; SB3 adds .zip automatically).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._model.save(path)

    @classmethod
    def load(cls, path: str, env) -> "DQNAgent":
        """
        Load a previously saved DQN model from disk.

        Args:
            path: Path to the .zip checkpoint produced by save().
            env: Environment instance to associate with the loaded model.

        Returns:
            DQNAgent instance with the loaded policy.
        """
        from stable_baselines3 import DQN

        instance = cls.__new__(cls)
        instance.env = env
        instance._model = DQN.load(path, env=env)
        return instance

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Expose underlying SB3 DQN model for callbacks / inspection."""
        return self._model
