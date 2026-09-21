"""
src/evaluation/experiment_runner.py — Full Research Experiment Suite

Orchestrates the complete Stage 8 experiment:

  For each scenario in [normal, high_volatility, low_liquidity, stress,
                        regime_transition, liquidity_shock]:
    For each seed in seeds:
      1. Generate synthetic market data (scenario-specific parameters)
      2. Chronological split: 70% train | 30% test
      3. Fit HMM on train split ONLY
      4. Evaluate TWAP, VWAP, POV on test data
      5. Train Model A (DQN, no regime) on train data, evaluate on test
      6. Train Model B (Regime-Aware DQN) on train data, evaluate on test
      7. Train Model C (Shuffled regime control) on train data, evaluate on test
      8. Collect all results into ResultsAggregator

Output:
  - results/experiment_results.csv    (raw records)
  - results/experiment_results.parquet
  - results/summary_table.csv         (mean/std per strategy)
  - results/comparison_table.csv      (wide format, all metrics)
  - results/per_scenario_table.csv    (strategy × scenario pivot)
  - results/rq_summary.json          (RQ1 / RQ2 / RQ3 delta answers)

All results written as-is. No post-hoc selection or modification.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.scenarios import generate_scenario_data, SCENARIOS
from src.evaluation.metrics import ResultsAggregator, SingleRunRecord

logger = logging.getLogger(__name__)


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_SCENARIOS = list(SCENARIOS.keys())
DEFAULT_SEEDS     = [42, 123, 777]    # 3 seeds for variability estimate
HORIZON_STEPS     = 30
TARGET_INVENTORY  = 100_000.0
SIDE              = "BUY"
RESULTS_DIR       = "results"


# ── Private helpers ────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame, ratio: float = 0.70):
    """Chronological train/test split. NEVER shuffle."""
    n = len(df)
    cut = int(n * ratio)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _fit_hmm(train_df: pd.DataFrame):
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler

    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(train_df)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)
    if len(X) < 10:
        return None  # not enough data to fit HMM for very small datasets

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42)
    try:
        hmm.fit(X, scaler=scaler)
        return hmm
    except Exception as e:
        logger.warning(f"HMM fitting failed: {e}")
        return None


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.regimes.features import RegimeFeatureEngine
    engine = RegimeFeatureEngine()
    return engine.compute_features(df)


def _eval_baselines(test_data: pd.DataFrame) -> List[Any]:
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner

    runner = BaselineRunner()
    results = []
    for strategy in [
        TWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        VWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        POVStrategy(target_inventory=TARGET_INVENTORY, target_rate=0.10),
    ]:
        r = runner.run(
            strategy=strategy,
            market_data=test_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )
        results.append(r)
    return results



def _train_and_eval_model_a(train_data, test_data, seed: int, timesteps: int, verbose: int):
    """Model A: DQN without regime."""
    from src.environment import TradeExecutionEnv
    from src.agents.dqn_agent import DQNAgent
    from src.agents.evaluator import evaluate_agent

    env = TradeExecutionEnv(
        market_data=train_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
    )
    agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=min(200, timesteps // 5))
    agent.train(total_timesteps=timesteps)

    return evaluate_agent(
        agent=agent,
        market_data=test_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        seed=seed,
        agent_name="DQN (no regime)",
    )


def _train_and_eval_model_b(train_data, test_data, hmm_model, seed: int, timesteps: int, verbose: int):
    """Model B: Regime-Aware DQN."""
    from src.environment import RegimeAwareTradeExecutionEnv
    from src.agents.dqn_agent import DQNAgent
    from src.agents.evaluator import evaluate_agent

    train_regime_feat = _compute_regime_features(train_data)
    env = RegimeAwareTradeExecutionEnv(
        hmm_model=hmm_model,
        regime_feature_data=train_regime_feat,
        market_data=train_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
    )
    agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=min(200, timesteps // 5))
    agent.train(total_timesteps=timesteps)

    return evaluate_agent(
        agent=agent,
        market_data=test_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        seed=seed,
        agent_name="Regime-Aware DQN",
    )


def _train_and_eval_model_c(train_data, test_data, seed: int, timesteps: int, verbose: int):
    """Model C: DQN with shuffled (uninformative) regime labels."""
    from src.evaluation.ablation import ShuffledRegimeEnv
    from src.agents.dqn_agent import DQNAgent
    from src.agents.evaluator import evaluate_agent

    env = ShuffledRegimeEnv(
        market_data=train_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        shuffle_seed=seed + 1000,
    )
    agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=min(200, timesteps // 5))
    agent.train(total_timesteps=timesteps)

    return evaluate_agent(
        agent=agent,
        market_data=test_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        seed=seed,
        agent_name="DQN (shuffled regime)",
    )


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_experiment_suite(
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    train_timesteps: int = 30_000,
    results_dir: str = RESULTS_DIR,
    verbose: int = 0,
    run_ablation: bool = True,
) -> ResultsAggregator:
    """
    Execute the full Stage 8 research experiment suite.

    Args:
        scenarios: List of scenario names to run (default: all 6).
        seeds: List of random seeds (default: [42, 123, 777]).
        train_timesteps: DQN training steps per model per seed.
        results_dir: Directory for output CSV/Parquet/JSON files.
        verbose: SB3 verbosity (0=quiet).
        run_ablation: Whether to include Model C (shuffled regime control).

    Returns:
        ResultsAggregator populated with all run records.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    seeds     = seeds     or DEFAULT_SEEDS

    aggregator = ResultsAggregator()
    os.makedirs(results_dir, exist_ok=True)

    total_runs = len(scenarios) * len(seeds)
    run_count  = 0
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("KAIRO — Stage 8: Full Research Experiment Suite")
    logger.info(f"  Scenarios : {scenarios}")
    logger.info(f"  Seeds     : {seeds}")
    logger.info(f"  Timesteps : {train_timesteps:,} per model")
    logger.info(f"  Total runs: {total_runs}")
    logger.info("=" * 70)

    for scenario_name in scenarios:
        for seed in seeds:
            run_count += 1
            logger.info(
                f"\n[{run_count}/{total_runs}] Scenario={scenario_name} | Seed={seed}"
            )

            # ── Generate data ────────────────────────────────────────────────
            all_data = generate_scenario_data(
                scenario_name=scenario_name, n_steps=200, seed=seed
            )
            train_data, test_data = _split(all_data, ratio=0.70)

            # ── Baselines (deterministic — run once per scenario/seed) ───────
            for baseline_result in _eval_baselines(test_data):
                aggregator.add(
                    SingleRunRecord.from_baseline_result(
                        baseline_result, scenario=scenario_name, seed=seed
                    )
                )

            # ── Fit HMM on train split only ──────────────────────────────────
            hmm_model = _fit_hmm(train_data)
            if hmm_model is None:
                logger.warning(f"  HMM failed for {scenario_name}/seed={seed}. Skipping DRL.")
                continue

            # ── Model A: DQN (no regime) ─────────────────────────────────────
            try:
                result_a = _train_and_eval_model_a(
                    train_data, test_data, seed, train_timesteps, verbose
                )
                aggregator.add(
                    SingleRunRecord.from_agent_eval_result(
                        result_a, scenario=scenario_name, seed=seed
                    )
                )
                logger.info(f"  Model A IS: {result_a.implementation_shortfall_bps:.2f} bps")
            except Exception as e:
                logger.error(f"  Model A failed: {e}")

            # ── Model B: Regime-Aware DQN ────────────────────────────────────
            try:
                result_b = _train_and_eval_model_b(
                    train_data, test_data, hmm_model, seed, train_timesteps, verbose
                )
                aggregator.add(
                    SingleRunRecord.from_agent_eval_result(
                        result_b, scenario=scenario_name, seed=seed
                    )
                )
                logger.info(f"  Model B IS: {result_b.implementation_shortfall_bps:.2f} bps")
            except Exception as e:
                logger.error(f"  Model B failed: {e}")

            # ── Model C: Shuffled regime control (ablation) ──────────────────
            if run_ablation:
                try:
                    result_c = _train_and_eval_model_c(
                        train_data, test_data, seed, train_timesteps, verbose
                    )
                    aggregator.add(
                        SingleRunRecord.from_agent_eval_result(
                            result_c, scenario=scenario_name, seed=seed
                        )
                    )
                    logger.info(f"  Model C IS: {result_c.implementation_shortfall_bps:.2f} bps")
                except Exception as e:
                    logger.error(f"  Model C failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"\nAll runs complete in {elapsed:.1f}s.")

    # ── Save results ───────────────────────────────────────────────────────────
    aggregator.save_csv(     os.path.join(results_dir, "experiment_results.csv"))
    aggregator.save_parquet( os.path.join(results_dir, "experiment_results.parquet"))

    summary = aggregator.summary_table()
    summary.to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)

    comparison = aggregator.comparison_table()
    comparison.to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)

    per_scenario = aggregator.per_scenario_table()
    per_scenario.to_csv(os.path.join(results_dir, "per_scenario_table.csv"), index=False)

    import json
    rq = aggregator.rq_summary()
    with open(os.path.join(results_dir, "rq_summary.json"), "w") as f:
        json.dump(rq, f, indent=2, default=str)

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("\n=== Summary Table (IS bps, mean ± std across seeds) ===")
    if not summary.empty:
        logger.info("\n" + summary.to_string(index=False))

    logger.info("\n=== Research Question Answers ===")
    for k, v in rq.items():
        logger.info(f"  {k}: {v:.3f}")

    return aggregator
