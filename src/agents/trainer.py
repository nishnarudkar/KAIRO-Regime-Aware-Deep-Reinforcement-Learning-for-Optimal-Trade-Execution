"""
Training pipeline for Stage 6 DQN Agent.

Responsibilities:
  - Build TradeExecutionEnv with synthetic or real market data
  - Train DQN via Stable-Baselines3
  - Log metrics to MLflow (experiment tracking)
  - Save model checkpoint to models/
  - Record full config + seed for reproducibility
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Full, self-contained training configuration.
    Stored alongside every MLflow run for exact reproducibility.
    """
    # Experiment identity
    experiment_name: str = "dqn_no_regime"
    run_name: str = ""                    # auto-filled if empty
    seed: int = 42

    # Training duration
    total_timesteps: int = 100_000

    # DQN hyperparameters
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net_arch: List[int] = field(default_factory=lambda: [128, 128])

    # Environment
    target_inventory: float = 100_000.0
    side: str = "BUY"
    horizon_steps: int = 30

    # Paths
    model_dir: str = "models"
    model_filename: str = "dqn_no_regime"

    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)


class DQNTrainer:
    """
    Orchestrates DQN training over a market data split.

    Data contract (chronological splits):
      - market_data must be the TRAINING split only.
      - Caller is responsible for ensuring no test-set rows are present.

    Args:
        config: TrainingConfig dataclass.
        market_data: Training-split DataFrame with columns
                     [timestamp, price, volume, spread, volatility].
        mlflow_tracking_uri: URI for MLflow tracking server (or local dir).
    """

    def __init__(
        self,
        config: TrainingConfig,
        market_data: Optional[pd.DataFrame] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.config = config
        self.market_data = market_data
        self.mlflow_tracking_uri = mlflow_tracking_uri or "mlruns"

    def _build_env(self):
        """Construct TradeExecutionEnv for training."""
        from src.environment import TradeExecutionEnv

        return TradeExecutionEnv(
            market_data=self.market_data,
            target_inventory=self.config.target_inventory,
            side=self.config.side,
            horizon_steps=self.config.horizon_steps,
            # With a long series, train on random windows instead of replaying one path.
            random_start=(self.market_data is not None and len(self.market_data) > 2 * self.config.horizon_steps),
        )

    def train(self) -> Dict[str, Any]:
        """
        Execute full training run.

        Returns:
            Dictionary of training metadata: run_id, model_path, duration_s.
        """
        import mlflow

        cfg = self.config

        # Auto-fill run name if missing
        if not cfg.run_name:
            cfg.run_name = f"dqn_seed{cfg.seed}_{int(time.time())}"

        # Seed global RNGs for full determinism
        np.random.seed(cfg.seed)

        from src.utils.dagshub_utils import setup_dagshub_mlflow
        dagshub_uri = setup_dagshub_mlflow()
        if dagshub_uri and self.mlflow_tracking_uri == "mlruns":
            self.mlflow_tracking_uri = dagshub_uri

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

        with mlflow.start_run(run_name=cfg.run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_id}")

            # Log all config params
            mlflow.log_params(asdict(cfg))

            # Build env and agent
            env = self._build_env()

            from src.agents.dqn_agent import DQNAgent

            agent = DQNAgent(
                env=env,
                seed=cfg.seed,
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                train_freq=cfg.train_freq,
                gradient_steps=cfg.gradient_steps,
                target_update_interval=cfg.target_update_interval,
                exploration_fraction=cfg.exploration_fraction,
                exploration_initial_eps=cfg.exploration_initial_eps,
                exploration_final_eps=cfg.exploration_final_eps,
                policy_kwargs={"net_arch": cfg.net_arch},
                verbose=1,
            )

            # Train
            start_time = time.time()
            logger.info(f"Training DQN for {cfg.total_timesteps:,} timesteps …")
            agent.train(total_timesteps=cfg.total_timesteps)
            duration_s = time.time() - start_time

            mlflow.log_metric("training_duration_s", duration_s)
            logger.info(f"Training complete in {duration_s:.1f}s")

            # Save checkpoint
            os.makedirs(cfg.model_dir, exist_ok=True)
            agent.save(cfg.model_path())
            mlflow.log_artifact(cfg.model_path() + ".zip", artifact_path="model")
            logger.info(f"Model saved → {cfg.model_path()}.zip")

            mlflow.log_param("run_id", run_id)

        return {
            "run_id": run_id,
            "model_path": cfg.model_path(),
            "duration_s": duration_s,
            "config": asdict(cfg),
        }
