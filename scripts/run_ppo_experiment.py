#!/usr/bin/env python

LEGACY DEMO: this script trains and evaluates on a single fixed 30-bar window of one short
series. It is kept as a quick smoke-level demonstration only. Research claims must come from
the unified suite (`scripts/run_experiments.py`, `src/evaluation/protocol.py`), which trains on
random windows and evaluates on many paired out-of-sample windows with statistics.
"""
scripts/run_ppo_experiment.py — Stage 9: Multi-Algorithm PPO Comparison

Trains 4 models and prints a head-to-head comparison table:
  Model A: DQN (no regime)
  Model B: Regime-Aware DQN
  Model C: PPO (no regime)
  Model D: Regime-Aware PPO

Research question addressed:
  RQ4: Is the regime-aware performance improvement specific to DQN,
       or does it generalise to PPO (on-policy policy-gradient)?

Usage
-----
    python scripts/run_ppo_experiment.py [options]

Examples
--------
    # Quick smoke test (5 000 steps)
    python scripts/run_ppo_experiment.py --timesteps 5000

    # Full research run
    python scripts/run_ppo_experiment.py --timesteps 100000

Flags
-----
--timesteps INT   Training steps per model (default: 50 000)
--seed      INT   Global random seed (default: 42)
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 9: Multi-Algorithm PPO Experiment")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed",      type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.agents.ppo_experiment import run_ppo_experiment

    results = run_ppo_experiment(
        train_timesteps=args.timesteps,
        seed=args.seed,
        verbose=0,
    )

    logger.info("Experiment complete.")
    for model_id, run_id in results["run_ids"].items():
        logger.info(f"  Model {model_id} MLflow run: {run_id}")
    logger.info("  View all runs: mlflow ui")


if __name__ == "__main__":
    main()
