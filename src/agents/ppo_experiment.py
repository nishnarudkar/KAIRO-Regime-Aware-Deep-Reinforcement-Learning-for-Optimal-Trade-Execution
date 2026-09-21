"""
src/agents/ppo_experiment.py — Stage 9: Multi-Algorithm A/B Experiment

Extends Stage 7's A/B experiment to a 4-model comparison:
  Model A: DQN (no regime) — original Stage 6/7 baseline
  Model B: Regime-Aware DQN — original Stage 7
  Model C: PPO (no regime) — new Stage 9
  Model D: Regime-Aware PPO — new Stage 9

Research Question Addressed:
  RQ4: Is the regime-aware performance improvement specific to DQN, or does
       it generalise to a second RL algorithm class (PPO)?

Decision Rules:
  - If Model D outperforms Model C (PPO gain from regime) similarly to how
    Model B outperformed Model A (DQN gain from regime):
      → Regime information benefit is ALGORITHM-CLASS-AGNOSTIC
  - If Model D ≈ Model C but Model B >> Model A:
      → Regime information benefit may be specific to DQN's value-function structure
  - Both outcomes are informative and neither is "wrong"

Experimental controls (identical across all 4 models):
  - Same synthetic market data + chronological split
  - Same action space (Discrete 4)
  - Same reward function + simulator
  - Same MLP architecture [128, 128]
  - Same random seed
  - gamma=0.99 for both algorithms
  - net_arch=[128,128] for both (explicit comparability)

All results logged to MLflow experiment "stage9_ppo_experiment".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EXPERIMENT_NAME  = "stage9_ppo_experiment"
SEED             = 42
HORIZON_STEPS    = 30
TARGET_INVENTORY = 100_000.0
SIDE             = "BUY"
DEFAULT_TRAIN_TIMESTEPS = 50_000


# ── Market data ────────────────────────────────────────────────────────────────

def _make_market_data(n: int = 300, seed: int = SEED) -> pd.DataFrame:
    """Deterministic synthetic OHLCV market data (same generator as Stage 7)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02 09:30", periods=n, freq="1min")
    close = 150.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + rng.uniform(0.02, 0.15, n)
    low  = close - rng.uniform(0.02, 0.15, n)
    volume = rng.integers(30_000, 80_000, n).astype(float)
    spread = rng.uniform(0.02, 0.06, n)
    volatility = rng.uniform(0.001, 0.003, n)
    return pd.DataFrame({
        "timestamp": dates,
        "open":  close, "high": high, "low": low, "close": close,
        "price": close, "volume": volume, "spread": spread, "volatility": volatility,
    })


def _compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.regimes.features import RegimeFeatureEngine
    return RegimeFeatureEngine().compute_features(df)


def _fit_hmm(train_df: pd.DataFrame):
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler
    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(train_df)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)
    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=SEED)
    hmm.fit(X, scaler=scaler)
    return hmm


def _train(agent_cls, env, timesteps: int, seed: int, verbose: int):
    agent = agent_cls(env=env, seed=seed, verbose=verbose, learning_starts=200
                      if hasattr(agent_cls, '__init__') and 'learning_starts' in
                      agent_cls.__init__.__code__.co_varnames else None)
    # PPO doesn't have learning_starts — create cleanly
    return agent


def _build_agent(agent_cls, env, seed: int, timesteps: int, verbose: int):
    """Instantiate the correct agent class and train it."""
    from src.agents.dqn_agent import DQNAgent
    from src.agents.ppo_agent import PPOAgent

    if agent_cls is DQNAgent:
        agent = DQNAgent(env=env, seed=seed, verbose=verbose, learning_starts=200)
    elif agent_cls is PPOAgent:
        agent = PPOAgent(env=env, seed=seed, verbose=verbose)
    else:
        agent = agent_cls(env=env, seed=seed, verbose=verbose)

    agent.train(total_timesteps=timesteps)
    return agent


def _eval(agent, test_data: pd.DataFrame, name: str):
    from src.agents.evaluator import evaluate_agent
    return evaluate_agent(
        agent=agent,
        market_data=test_data,
        target_inventory=TARGET_INVENTORY,
        side=SIDE,
        horizon_steps=HORIZON_STEPS,
        seed=SEED,
        agent_name=name,
    )


def _eval_baselines(test_data: pd.DataFrame) -> list:
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner
    runner = BaselineRunner()
    return [
        runner.run(
            strategy=s,
            market_data=test_data,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )
        for s in [
            TWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
            VWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
            POVStrategy(target_inventory=TARGET_INVENTORY, target_rate=0.10),
        ]
    ]


def _print_table(results: list) -> None:
    header = (
        f"{'Model':<28} {'IS (bps)':>10} {'Cost ($)':>10} "
        f"{'Fill %':>8} {'VWAP slip':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        name = getattr(r, "strategy_name", None) or getattr(r, "agent_name", "?")
        print(
            f"{name:<28} "
            f"{r.implementation_shortfall_bps:>10.2f} "
            f"{r.execution_cost:>10.2f} "
            f"{r.completion_rate * 100:>7.1f}% "
            f"{r.vwap_slippage_bps:>10.2f}"
        )
    print(sep + "\n")


# ── Main experiment ────────────────────────────────────────────────────────────

def run_ppo_experiment(
    train_timesteps: int = DEFAULT_TRAIN_TIMESTEPS,
    seed: int = SEED,
    mlflow_tracking_uri: str = "mlruns",
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Execute Stage 9: 4-model multi-algorithm comparison.

    Models:
      A: DQN (no regime)
      B: Regime-Aware DQN
      C: PPO (no regime)
      D: Regime-Aware PPO

    Returns dict with all results and MLflow run IDs.
    """
    import os
    import mlflow
    from src.agents.dqn_agent import DQNAgent
    from src.agents.ppo_agent import PPOAgent
    from src.environment import TradeExecutionEnv, RegimeAwareTradeExecutionEnv

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=" * 65)
    logger.info("KAIRO — Stage 9: Multi-Algorithm PPO Experiment")
    logger.info(f"  Timesteps : {train_timesteps:,}")
    logger.info(f"  Seed      : {seed}")
    logger.info("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────────
    all_data = _make_market_data(n=300, seed=seed)
    split_idx = int(len(all_data) * 0.70)
    train_data = all_data.iloc[:split_idx].reset_index(drop=True)
    test_data  = all_data.iloc[split_idx:].reset_index(drop=True)

    # ── HMM fitted on train split ONLY ────────────────────────────────────────
    hmm = _fit_hmm(train_data)
    train_regime_feat = _compute_regime_features(train_data)
    test_regime_feat  = _compute_regime_features(test_data)

    # ── Baselines ──────────────────────────────────────────────────────────────
    baselines = _eval_baselines(test_data)

    results = {}
    run_ids = {}

    configs = [
        ("A", "DQN (no regime)",      DQNAgent, False),
        ("B", "Regime-Aware DQN",     DQNAgent, True),
        ("C", "PPO (no regime)",      PPOAgent, False),
        ("D", "Regime-Aware PPO",     PPOAgent, True),
    ]

    for model_id, model_name, AgentCls, use_regime in configs:
        logger.info(f"Training Model {model_id}: {model_name} …")

        with mlflow.start_run(run_name=f"Model{model_id}_{model_name.replace(' ','_')}_seed{seed}") as run:
            mlflow.log_params({
                "model_id": model_id,
                "algorithm": AgentCls.__name__,
                "regime_in_state": use_regime,
                "seed": seed,
                "train_timesteps": train_timesteps,
                "net_arch": "[128, 128]",
                "gamma": 0.99,
            })

            if use_regime:
                env = RegimeAwareTradeExecutionEnv(
                    hmm_model=hmm,
                    regime_feature_data=train_regime_feat,
                    market_data=train_data,
                    target_inventory=TARGET_INVENTORY,
                    side=SIDE,
                    horizon_steps=HORIZON_STEPS,
                )
                eval_env = RegimeAwareTradeExecutionEnv(
                    hmm_model=hmm,
                    regime_feature_data=test_regime_feat,
                    market_data=test_data,
                    target_inventory=TARGET_INVENTORY,
                    side=SIDE,
                    horizon_steps=HORIZON_STEPS,
                )
            else:
                env = TradeExecutionEnv(
                    market_data=train_data,
                    target_inventory=TARGET_INVENTORY,
                    side=SIDE,
                    horizon_steps=HORIZON_STEPS,
                )
                eval_env = TradeExecutionEnv(
                    market_data=test_data,
                    target_inventory=TARGET_INVENTORY,
                    side=SIDE,
                    horizon_steps=HORIZON_STEPS,
                )

            t0 = time.time()
            agent = _build_agent(AgentCls, env, seed=seed, timesteps=train_timesteps, verbose=verbose)
            mlflow.log_metric("train_duration_s", time.time() - t0)

            from src.agents.evaluator import evaluate_agent
            result = evaluate_agent(
                agent=agent,
                market_data=test_data,
                target_inventory=TARGET_INVENTORY,
                side=SIDE,
                horizon_steps=HORIZON_STEPS,
                seed=seed,
                agent_name=model_name,
                env=eval_env,
            )
            mlflow.log_metrics({
                "is_bps":            result.implementation_shortfall_bps,
                "execution_cost":    result.execution_cost,
                "completion_rate":   result.completion_rate,
                "vwap_slippage_bps": result.vwap_slippage_bps,
                "terminal_penalty":  result.terminal_penalty,
            })

            results[model_id] = result
            run_ids[model_id] = run.info.run_id
            logger.info(f"  IS={result.implementation_shortfall_bps:.2f} bps")

    # ── Results table ──────────────────────────────────────────────────────────
    all_results = baselines + list(results.values())
    print("\n=== KAIRO Stage 9 - Multi-Algorithm Comparison ===")
    _print_table(all_results)

    # ── RQ4 answer ─────────────────────────────────────────────────────────────
    dqn_gain  = results["A"].implementation_shortfall_bps - results["B"].implementation_shortfall_bps
    ppo_gain  = results["C"].implementation_shortfall_bps - results["D"].implementation_shortfall_bps

    print("=== Research Question Answers (preliminary - short training run) ===")
    print(f"  RQ2 (DQN): Regime DQN vs plain DQN IS improvement: {dqn_gain:+.2f} bps")
    print(f"  RQ4 (PPO): Regime PPO vs plain PPO IS improvement: {ppo_gain:+.2f} bps")
    if (dqn_gain > 0) and (ppo_gain > 0):
        print("  -> Evidence: Regime benefit appears ACROSS BOTH algorithm classes.")
    elif (dqn_gain > 0) and (ppo_gain <= 0):
        print("  -> Evidence: Regime benefit may be specific to DQN (not replicated by PPO).")
    elif (dqn_gain <= 0) and (ppo_gain > 0):
        print("  -> Evidence: Regime benefit only observed in PPO.")
    else:
        print("  -> No regime benefit observed in either algorithm (short run - needs more timesteps).")
    print("  NOTE: Interpret with caution. Use --timesteps 100000+ for research results.\n")

    return {
        "run_ids": run_ids,
        "baselines": baselines,
        "results": results,
        "rq2_dqn_gain_bps": dqn_gain,
        "rq4_ppo_gain_bps": ppo_gain,
    }
