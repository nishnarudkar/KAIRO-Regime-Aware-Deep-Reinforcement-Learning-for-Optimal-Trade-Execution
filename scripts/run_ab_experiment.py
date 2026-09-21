#!/usr/bin/env python
"""
scripts/run_ab_experiment.py — Stage 7 Regime-Aware DQN A/B Experiment

Trains Model A (DQN, no regime) and Model B (Regime-Aware DQN) under
IDENTICAL experimental conditions, then prints a head-to-head comparison.

Usage
-----
    python scripts/run_ab_experiment.py [options]

Examples
--------
    # Quick smoke test (10 000 steps)
    python scripts/run_ab_experiment.py --timesteps 10000

    # Full research run (100 000 steps)
    python scripts/run_ab_experiment.py --timesteps 100000

    # Custom seed
    python scripts/run_ab_experiment.py --timesteps 50000 --seed 7

Flags
-----
--timesteps INT   Training steps per model    (default: 50 000)
--seed      INT   Global random seed          (default: 42)
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
    parser = argparse.ArgumentParser(description="Stage 7: A/B Regime-Aware DQN Experiment")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed",      type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("KAIRO — Stage 7: Regime-Aware DQN A/B Experiment")
    logger.info(f"  Timesteps per model : {args.timesteps:,}")
    logger.info(f"  Seed                : {args.seed}")
    logger.info("=" * 65)

    from src.agents.ab_experiment import run_ab_experiment

    results = run_ab_experiment(
        train_timesteps=args.timesteps,
        seed=args.seed,
        verbose=0,
    )

    logger.info("Experiment complete.")
    logger.info(f"  MLflow Model A run : {results['run_id_a']}")
    logger.info(f"  MLflow Model B run : {results['run_id_b']}")
    logger.info("  View all runs      : mlflow ui")


if __name__ == "__main__":
    main()
