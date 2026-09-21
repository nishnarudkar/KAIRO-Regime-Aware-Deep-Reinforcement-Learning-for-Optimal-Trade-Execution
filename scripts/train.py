#!/usr/bin/env python
"""
scripts/train.py — Stage 6 DQN Training Script

Usage
-----
Activate the venv, then from the repo root:

    python scripts/train.py [options]

Examples
--------
    # Quick smoke test (5 000 steps, synthetic data)
    python scripts/train.py --timesteps 5000 --seed 0

    # Full run (100 000 steps, seed 42)
    python scripts/train.py --timesteps 100000 --seed 42

    # Custom experiment name for MLflow
    python scripts/train.py --experiment my_run --timesteps 50000

Flags
-----
--timesteps INT   Total environment steps to train (default: 100 000)
--seed      INT   Global random seed              (default: 42)
--experiment STR  MLflow experiment name          (default: dqn_no_regime)
--run-name  STR   MLflow run name                 (auto if empty)
--lr        FLOAT Adam learning rate              (default: 1e-4)
--gamma     FLOAT Discount factor                 (default: 0.99)
--model-dir STR   Directory to save checkpoints   (default: models)
--horizon   INT   Episode horizon steps           (default: 30)
"""

import argparse
import logging
import sys
import os

# Allow imports from repo root regardless of invocation directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage-6 DQN agent (no regime information)"
    )
    parser.add_argument("--timesteps",  type=int,   default=100_000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--experiment", type=str,   default="dqn_no_regime")
    parser.add_argument("--run-name",   type=str,   default="")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--model-dir",  type=str,   default="models")
    parser.add_argument("--horizon",    type=int,   default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.agents.trainer import DQNTrainer, TrainingConfig

    cfg = TrainingConfig(
        experiment_name=args.experiment,
        run_name=args.run_name,
        seed=args.seed,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        model_dir=args.model_dir,
        horizon_steps=args.horizon,
    )

    logger.info("=" * 60)
    logger.info("KAIRO — Stage 6: DQN Training (no regime)")
    logger.info(f"  Timesteps : {cfg.total_timesteps:,}")
    logger.info(f"  Seed      : {cfg.seed}")
    logger.info(f"  Experiment: {cfg.experiment_name}")
    logger.info(f"  Model dir : {cfg.model_dir}")
    logger.info("=" * 60)

    # Synthetic regime-switching series (train region only); episodes start at random windows.
    from src.evaluation.scenarios import generate_scenario_data
    series = generate_scenario_data("normal", n_steps=3000, seed=args.seed)
    trainer = DQNTrainer(config=cfg, market_data=series.iloc[:2100].reset_index(drop=True))
    result = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  MLflow Run ID : {result['run_id']}")
    logger.info(f"  Model saved   : {result['model_path']}.zip")
    logger.info(f"  Duration      : {result['duration_s']:.1f}s")
    logger.info("=" * 60)
    logger.info("Next step: run  python scripts/evaluate.py  to benchmark vs baselines.")


if __name__ == "__main__":
    main()
