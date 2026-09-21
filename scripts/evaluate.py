#!/usr/bin/env python
"""
scripts/evaluate.py — Stage 6 DQN Evaluation + Baseline Comparison

Loads a trained DQN checkpoint and evaluates it against TWAP, VWAP, and POV
on the same synthetic market data, printing a side-by-side comparison table.

Usage
-----
    python scripts/evaluate.py [options]

Examples
--------
    # Evaluate the default checkpoint (models/dqn_no_regime.zip)
    python scripts/evaluate.py

    # Evaluate a custom checkpoint path
    python scripts/evaluate.py --model models/my_run

    # Specify seed and horizon
    python scripts/evaluate.py --seed 0 --horizon 30

Flags
-----
--model     STR   Path to .zip checkpoint (without extension)
--seed      INT   Random seed for evaluation episode   (default: 42)
--horizon   INT   Episode horizon steps                (default: 30)
--inventory FLOAT Target shares to execute             (default: 100 000)
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
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-6 DQN vs TWAP / VWAP / POV"
    )
    parser.add_argument("--model",     type=str,   default="models/dqn_no_regime")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--horizon",   type=int,   default=30)
    parser.add_argument("--inventory", type=float, default=100_000.0)
    return parser.parse_args()


def _build_synthetic_data(horizon: int):
    """Build identical deterministic market data for fair comparison."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01 09:30", periods=horizon, freq="1min")
    prices = 150.0 + np.cumsum(rng.normal(0, 0.05, horizon))
    volumes = rng.integers(30_000, 80_000, horizon).astype(float)
    spreads = rng.uniform(0.02, 0.06, horizon)
    vols = rng.uniform(0.001, 0.003, horizon)

    return pd.DataFrame({
        "timestamp": dates,
        "price": prices,
        "volume": volumes,
        "spread": spreads,
        "volatility": vols,
    })


def _print_comparison(results: list):
    """Pretty-print comparison table."""
    header = (
        f"{'Strategy':<22} {'IS (bps)':>10} {'Cost ($)':>10} "
        f"{'Fill %':>8} {'VWAP slip (bps)':>16} {'Penalty ($)':>12}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for r in results:
        name = getattr(r, "strategy_name", None) or getattr(r, "agent_name", "?")
        print(
            f"{name:<22} "
            f"{r.implementation_shortfall_bps:>10.2f} "
            f"{r.execution_cost:>10.2f} "
            f"{r.completion_rate * 100:>7.1f}% "
            f"{r.vwap_slippage_bps:>16.2f} "
            f"{r.terminal_penalty:>12.2f}"
        )
    print(separator + "\n")


def main():
    args = parse_args()

    market_data = _build_synthetic_data(args.horizon)
    all_results = []

    # ── Baselines ─────────────────────────────────────────────────────────
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner

    runner = BaselineRunner()
    for strategy in [TWAPStrategy(target_inventory=args.inventory, total_steps=args.horizon), VWAPStrategy(target_inventory=args.inventory, total_steps=args.horizon), POVStrategy(target_inventory=args.inventory, target_rate=0.10)]:
        result = runner.run(
            strategy=strategy,
            market_data=market_data,
            target_inventory=args.inventory,
            side="BUY",
            horizon_steps=args.horizon,
        )
        all_results.append(result)
        logger.info(f"Baseline [{result.strategy_name}] evaluated.")

    # ── DQN Agent ─────────────────────────────────────────────────────────
    model_zip = args.model + ".zip"
    if not os.path.exists(model_zip):
        logger.warning(
            f"Model not found at {model_zip}. "
            "Run  python scripts/train.py  first to create a checkpoint."
        )
    else:
        from src.environment import TradeExecutionEnv
        from src.agents.dqn_agent import DQNAgent
        from src.agents.evaluator import evaluate_agent

        env = TradeExecutionEnv(
            market_data=market_data,
            target_inventory=args.inventory,
            side="BUY",
            horizon_steps=args.horizon,
        )
        agent = DQNAgent.load(args.model, env=env)

        dqn_result = evaluate_agent(
            agent=agent,
            market_data=market_data,
            target_inventory=args.inventory,
            horizon_steps=args.horizon,
            seed=args.seed,
            agent_name="DQN (no regime)",
        )
        all_results.append(dqn_result)
        logger.info("DQN evaluation complete.")

        logger.info(
            f"DQN action distribution: "
            + ", ".join(
                f"{dqn_result.action_labels.get(k, k)}={v}"
                for k, v in sorted(dqn_result.action_counts.items())
            )
        )

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n=== KAIRO Stage 6 — Execution Strategy Comparison ===")
    _print_comparison(all_results)

    logger.info("Evaluation complete. Inspect MLflow at: mlflow ui")


if __name__ == "__main__":
    main()
