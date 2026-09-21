#!/usr/bin/env python
"""
scripts/run_experiments.py — Stage 8: Full Research Experiment Suite

Runs all strategies across all scenarios and seeds, writes results to results/.

Usage
-----
    python scripts/run_experiments.py [options]

Examples
--------
    # Quick smoke test: 1 scenario, 1 seed, 5 000 steps
    python scripts/run_experiments.py --scenarios normal --seeds 42 --timesteps 5000

    # Full suite (slow): all 6 scenarios, 3 seeds, 50 000 steps each
    python scripts/run_experiments.py --timesteps 50000

    # Without ablation (faster)
    python scripts/run_experiments.py --no-ablation --timesteps 20000

Flags
-----
--scenarios  STR [STR ...]  Scenario names (default: all 6)
--seeds      INT [INT ...]  Random seeds   (default: 42 123 777)
--timesteps  INT            DQN train steps per model (default: 30 000)
--no-ablation               Skip Model C shuffled-regime control
--results-dir STR           Output directory (default: results)
"""

import os

# One thread per worker process: without this every worker spawns a BLAS thread pool
# sized to the machine and parallel runs exhaust memory. Must be set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    from src.evaluation.scenarios import SCENARIOS
    parser = argparse.ArgumentParser(description="Stage 8 Research Experiment Suite")
    parser.add_argument("--scenarios",   nargs="+", default=list(SCENARIOS.keys()))
    parser.add_argument("--seeds",       nargs="+", type=int, default=[42, 123, 777, 2024, 31415])
    parser.add_argument("--timesteps",   type=int,  default=60_000)
    parser.add_argument("--jobs",        type=int,  default=1, help="parallel worker processes")
    parser.add_argument("--n-bars",      type=int,  default=3000)
    parser.add_argument("--horizon",     type=int,  default=30, help="execution horizon in one-minute bars")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-ppo",      action="store_true")
    parser.add_argument("--results-dir", type=str,  default="results")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("KAIRO — Stage 8: Research Experiment Suite")
    logger.info(f"  Scenarios : {args.scenarios}")
    logger.info(f"  Seeds     : {args.seeds}")
    logger.info(f"  Timesteps : {args.timesteps:,} per model")
    logger.info(f"  Ablation  : {'disabled' if args.no_ablation else 'enabled (Model C)'}")
    logger.info(f"  Output    : {args.results_dir}/")
    logger.info("=" * 70)

    from src.evaluation.experiment_runner import run_experiment_suite

    aggregator = run_experiment_suite(
        scenarios=args.scenarios,
        seeds=args.seeds,
        train_timesteps=args.timesteps,
        results_dir=args.results_dir,
        verbose=0,
        run_ablation=not args.no_ablation,
        include_ppo=not args.no_ppo,
        n_bars=args.n_bars,
        n_jobs=args.jobs,
        horizon=args.horizon,
    )

    logger.info("=" * 70)
    logger.info("Experiment suite complete.")
    logger.info(f"  Results written to: {os.path.abspath(args.results_dir)}/")
    logger.info("  Files: experiment_results.csv, summary_table.csv,")
    logger.info("         comparison_table.csv, per_scenario_table.csv, paired_comparisons.csv,")
    logger.info("         window_results.csv, hmm_validation.csv, rq_summary.json, run_config.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
