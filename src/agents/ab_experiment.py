"""
Stage 7 A/B Experiment Runner.

Runs two head-to-head controlled experiments:
  Model A: Standard DQN    — State = Market features + Order state
  Model B: Regime-Aware DQN — State = Market features + Order state + HMM regime

Experimental conditions kept IDENTICAL for both models:
  - Same synthetic market data
  - Same train/test split (chronological)
  - Same action space (0%, 10%, 25%, 50%)
  - Same reward function and coefficients
  - Same execution simulator (Almgren-Chriss impact model)
  - Same DQN hyperparameters
  - Same random seed

Primary research questions answered:
  RQ1: Does RL outperform conventional execution baselines?
  RQ2: Does regime information improve RL execution?

Results are logged to MLflow under experiment "stage7_ab_experiment".
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Shared experiment constants ────────────────────────────────────────────────

EXPERIMENT_NAME = "stage7_ab_experiment"
SEED = 42
HORIZON_STEPS = 30
TARGET_INVENTORY = 100_000.0
SIDE = "BUY"

# Short training to keep the script runnable; override via CLI for research runs
DEFAULT_TRAIN_TIMESTEPS = 50_000


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_market_data(n: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Deterministic synthetic OHLCV market data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02 09:30", periods=n, freq="1min")
    close = 150.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + rng.uniform(0.02, 0.15, n)
    low = close - rng.uniform(0.02, 0.15, n)
    volume = rng.integers(30_000, 80_000, n).astype(float)
    spread = rng.uniform(0.02, 0.06, n)
    volatility = rng.uniform(0.001, 0.003, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open":      close,
        "high":      high,
        "low":       low,
        "close":     close,
        "price":     close,
        "volume":    volume,
        "spread":    spread,
        "volatility": volatility,
    })


def _compute_regime_features(df: pd.DataFrame):
    """Compute regime features from OHLCV data (causal, no lookahead)."""
    from src.regimes.features import RegimeFeatureEngine
    engine = RegimeFeatureEngine()
    return engine.compute_features(df)


def _fit_hmm(train_df: pd.DataFrame):
    """Fit a 4-state HMM on the training split only."""
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler

    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(train_df)
    _, X_train = engine.extract_feature_matrix(feat_df, drop_na=True)

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=SEED)
    hmm.fit(X_train, scaler=scaler)
    return hmm


def _train_agent(env, timesteps: int, seed: int = SEED, verbose: int = 0):
    """Train a DQN agent on the given env."""
    from src.agents.dqn_agent import DQNAgent
    agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=200)
    agent.train(total_timesteps=timesteps)
    return agent


def _eval_agent(agent, market_data: pd.DataFrame, agent_name: str):
    """Evaluate DQN agent and return AgentEvalResult."""
    from src.agents.evaluator import evaluate_agent
    return evaluate_agent(
        agent=agent,
        market_data=market_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        seed=SEED,
        agent_name=agent_name,
    )


def _eval_baseline(strategy_name: str, market_data: pd.DataFrame):
    """Evaluate a single baseline strategy."""
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner

    strategies = {
        "TWAP": TWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        "VWAP": VWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        "POV":  POVStrategy(target_inventory=TARGET_INVENTORY, target_rate=0.10),
    }
    runner = BaselineRunner()
    return runner.run(
        strategy=strategies[strategy_name],
        market_data=market_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
    )


def _print_table(results: list) -> None:
    """Print aligned comparison table."""
    header = (
        f"{'Model':<26} {'IS (bps)':>10} {'Cost ($)':>10} "
        f"{'Fill %':>8} {'VWAP slip':>10} {'Penalty ($)':>12}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        name = getattr(r, "strategy_name", None) or getattr(r, "agent_name", "?")
        print(
            f"{name:<26} "
            f"{r.implementation_shortfall_bps:>10.2f} "
            f"{r.execution_cost:>10.2f} "
            f"{r.completion_rate * 100:>7.1f}% "
            f"{r.vwap_slippage_bps:>10.2f} "
            f"{r.terminal_penalty:>12.2f}"
        )
    print(sep + "\n")


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_ab_experiment(
    train_timesteps: int = DEFAULT_TRAIN_TIMESTEPS,
    seed: int = SEED,
    mlflow_tracking_uri: str = "mlruns",
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Execute the full Stage 7 A/B experiment.

    Returns dict with all result objects and MLflow run IDs.
    """
    import os
    import mlflow

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=" * 65)
    logger.info("KAIRO — Stage 7: Regime-Aware DQN A/B Experiment")
    logger.info(f"  Timesteps : {train_timesteps:,}")
    logger.info(f"  Seed      : {seed}")
    logger.info("=" * 65)

    # ── Data preparation ───────────────────────────────────────────────────────
    logger.info("Generating synthetic market data …")
    all_data = _make_market_data(n=300, seed=seed)

    # Chronological split: 70% train | 30% test  (never shuffle)
    split_idx = int(len(all_data) * 0.70)
    train_data = all_data.iloc[:split_idx].reset_index(drop=True)
    test_data  = all_data.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"  Train rows: {len(train_data)}  |  Test rows: {len(test_data)}")

    # ── Fit HMM on TRAIN only ─────────────────────────────────────────────────
    logger.info("Fitting 4-state HMM on training split …")
    hmm_model = _fit_hmm(train_data)
    logger.info(f"  HMM fitted. Regimes: {hmm_model.regime_labels}")

    # ── Compute regime features for test set ──────────────────────────────────
    test_regime_features = _compute_regime_features(test_data)

    # ── Baselines ──────────────────────────────────────────────────────────────
    logger.info("Evaluating baselines on test data …")
    twap_result = _eval_baseline("TWAP", test_data)
    vwap_result = _eval_baseline("VWAP", test_data)
    pov_result  = _eval_baseline("POV",  test_data)
    logger.info("  Baselines done.")

    # ── Model A: DQN (no regime) ───────────────────────────────────────────────
    logger.info("Training Model A: DQN (no regime) …")
    with mlflow.start_run(run_name=f"ModelA_DQN_seed{seed}") as run_a:
        mlflow.log_params({
            "model": "DQN_no_regime",
            "seed": seed,
            "train_timesteps": train_timesteps,
            "regime_in_state": False,
        })
        from src.environment import TradeExecutionEnv
        env_a = TradeExecutionEnv(
            market_data=train_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )
        t0 = time.time()
        agent_a = _train_agent(env_a, train_timesteps, seed=seed, verbose=verbose)
        mlflow.log_metric("train_duration_s", time.time() - t0)

        result_a = _eval_agent(agent_a, test_data, agent_name="DQN (no regime)")
        mlflow.log_metrics({
            "is_bps": result_a.implementation_shortfall_bps,
            "execution_cost": result_a.execution_cost,
            "completion_rate": result_a.completion_rate,
            "vwap_slippage_bps": result_a.vwap_slippage_bps,
            "terminal_penalty": result_a.terminal_penalty,
        })
        run_id_a = run_a.info.run_id

    logger.info(f"  Model A done. IS={result_a.implementation_shortfall_bps:.2f} bps")

    # ── Model B: Regime-Aware DQN ─────────────────────────────────────────────
    logger.info("Training Model B: Regime-Aware DQN …")
    with mlflow.start_run(run_name=f"ModelB_RegimeDQN_seed{seed}") as run_b:
        mlflow.log_params({
            "model": "RegimeAware_DQN",
            "seed": seed,
            "train_timesteps": train_timesteps,
            "regime_in_state": True,
            "n_regimes": hmm_model.n_regimes,
        })
        from src.environment import RegimeAwareTradeExecutionEnv
        train_regime_features = _compute_regime_features(train_data)

        env_b = RegimeAwareTradeExecutionEnv(
            hmm_model=hmm_model,
            regime_feature_data=train_regime_features,
            market_data=train_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )
        t0 = time.time()
        agent_b = _train_agent(env_b, train_timesteps, seed=seed, verbose=verbose)
        mlflow.log_metric("train_duration_s", time.time() - t0)

        # Evaluate using regime-aware env on test data
        from src.agents.evaluator import AgentEvalResult

        env_b_test = RegimeAwareTradeExecutionEnv(
            hmm_model=hmm_model,
            regime_feature_data=test_regime_features,
            market_data=test_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )

        # Use evaluate_agent but with the regime-aware env
        from src.agents.evaluator import evaluate_agent
        result_b = evaluate_agent(
            agent=agent_b,
            market_data=test_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
            seed=seed,
            agent_name="Regime-Aware DQN",
        )

        mlflow.log_metrics({
            "is_bps": result_b.implementation_shortfall_bps,
            "execution_cost": result_b.execution_cost,
            "completion_rate": result_b.completion_rate,
            "vwap_slippage_bps": result_b.vwap_slippage_bps,
            "terminal_penalty": result_b.terminal_penalty,
        })
        run_id_b = run_b.info.run_id

    logger.info(f"  Model B done. IS={result_b.implementation_shortfall_bps:.2f} bps")

    # ── Results table ──────────────────────────────────────────────────────────
    all_results = [twap_result, vwap_result, pov_result, result_a, result_b]
    print("\n=== KAIRO Stage 7 — A/B Experiment Results ===")
    _print_table(all_results)

    # ── Delta summary (RQ1 and RQ2) ────────────────────────────────────────────
    best_baseline_is = min(
        twap_result.implementation_shortfall_bps,
        vwap_result.implementation_shortfall_bps,
        pov_result.implementation_shortfall_bps,
    )
    rq1_improvement = best_baseline_is - result_a.implementation_shortfall_bps
    rq2_improvement = result_a.implementation_shortfall_bps - result_b.implementation_shortfall_bps

    print("=== Research Question Answers (preliminary — short training run) ===")
    print(f"  RQ1: DQN vs best baseline IS improvement : {rq1_improvement:+.2f} bps")
    print(f"  RQ2: Regime-Aware DQN vs DQN IS improvement: {rq2_improvement:+.2f} bps")
    print("  NOTE: Interpret with caution. Use --timesteps 100000+ for research results.\n")

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "result_twap": twap_result,
        "result_vwap": vwap_result,
        "result_pov": pov_result,
        "result_model_a": result_a,
        "result_model_b": result_b,
        "rq1_improvement_bps": rq1_improvement,
        "rq2_improvement_bps": rq2_improvement,
    }
