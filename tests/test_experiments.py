"""
tests/test_experiments.py — Unit tests for Stage 8 Experiment Suite

Tests cover:
  - Scenario data generators (all 6 scenarios)
  - ResultsAggregator / SingleRunRecord
  - ShuffledRegimeEnv (Model C ablation)
  - Experiment runner smoke test (minimal timesteps)
  - Output file writing

All tests use synthetic data and minimal DRL training (smoke tests only).
No live API calls. Seeds are fixed.
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.scenarios import generate_scenario_data, SCENARIOS


# ── Scenario generation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_scenario_generates_valid_dataframe(scenario_name):
    """Every scenario must produce a DataFrame with required columns."""
    df = generate_scenario_data(scenario_name, n_steps=50, seed=0)
    required = {"timestamp", "price", "volume", "spread", "volatility",
                "open", "high", "low", "close"}
    assert required.issubset(set(df.columns)), (
        f"Scenario '{scenario_name}' missing columns: {required - set(df.columns)}"
    )
    assert len(df) == 50
    assert df["price"].isna().sum() == 0
    assert (df["volume"] > 0).all()


def test_scenario_prices_positive():
    for name in SCENARIOS:
        df = generate_scenario_data(name, n_steps=30, seed=1)
        assert (df["price"] > 0).all(), f"Negative price in scenario {name}"


def test_scenario_high_ge_low():
    for name in SCENARIOS:
        df = generate_scenario_data(name, n_steps=30, seed=2)
        assert (df["high"] >= df["low"]).all(), f"high < low in scenario {name}"


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        generate_scenario_data("nonexistent_scenario", n_steps=10, seed=0)


def test_scenario_deterministic():
    """Same seed must produce identical data."""
    df1 = generate_scenario_data("normal", n_steps=30, seed=99)
    df2 = generate_scenario_data("normal", n_steps=30, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_different_data():
    df1 = generate_scenario_data("normal", n_steps=30, seed=0)
    df2 = generate_scenario_data("normal", n_steps=30, seed=1)
    assert not df1["price"].equals(df2["price"])


# ── Metrics aggregation ────────────────────────────────────────────────────────

def _make_dummy_record(strategy="TWAP", scenario="normal", seed=42, is_bps=10.0):
    from src.evaluation.metrics import SingleRunRecord
    return SingleRunRecord(
        strategy_name=strategy,
        scenario=scenario,
        seed=seed,
        implementation_shortfall_bps=is_bps,
        execution_cost=100.0,
        market_impact_cost=50.0,
        total_transaction_fees=5.0,
        completion_rate=1.0,
        average_execution_price=150.0,
        arrival_price=149.5,
        vwap_slippage_bps=2.0,
        terminal_penalty=0.0,
        turnover=15_000_000.0,
    )


def test_aggregator_collects_records():
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    agg.add(_make_dummy_record("TWAP", "normal", 42, 15.0))
    agg.add(_make_dummy_record("DQN (no regime)", "normal", 42, 10.0))
    df = agg.to_dataframe()
    assert len(df) == 2
    assert set(df["strategy_name"]) == {"TWAP", "DQN (no regime)"}


def test_summary_table_sorts_by_mean():
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    agg.add(_make_dummy_record("TWAP",    is_bps=20.0))
    agg.add(_make_dummy_record("DQN (no regime)", is_bps=8.0))
    summary = agg.summary_table(metric="implementation_shortfall_bps")
    assert summary.iloc[0]["strategy"] == "DQN (no regime)"


def test_comparison_table_shape():
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    for name in ["TWAP", "VWAP", "POV"]:
        agg.add(_make_dummy_record(name))
    tbl = agg.comparison_table()
    assert "strategy" in tbl.columns
    assert "implementation_shortfall_bps" in tbl.columns
    assert len(tbl) == 3


def test_per_scenario_pivot():
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    agg.add(_make_dummy_record("TWAP", "normal",         is_bps=10.0))
    agg.add(_make_dummy_record("TWAP", "high_volatility", is_bps=20.0))
    pivot = agg.per_scenario_table()
    assert "normal" in pivot.columns
    assert "high_volatility" in pivot.columns


def test_rq_summary_returns_dict():
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    for name in ["TWAP", "VWAP", "POV"]:
        agg.add(_make_dummy_record(name, is_bps=15.0))
    agg.add(_make_dummy_record("DQN (no regime)", is_bps=10.0))
    agg.add(_make_dummy_record("Regime-Aware DQN", is_bps=7.0))
    rq = agg.rq_summary()
    assert "rq1_dqn_vs_baseline_bps" in rq
    assert "rq2_regime_vs_plain_dqn_bps" in rq
    assert rq["rq1_dqn_vs_baseline_bps"] == pytest.approx(5.0)
    assert rq["rq2_regime_vs_plain_dqn_bps"] == pytest.approx(3.0)


def test_aggregator_saves_csv(tmp_path):
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    agg.add(_make_dummy_record())
    path = str(tmp_path / "test.csv")
    agg.save_csv(path)
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 1


def test_aggregator_saves_json(tmp_path):
    from src.evaluation.metrics import ResultsAggregator
    agg = ResultsAggregator()
    agg.add(_make_dummy_record())
    path = str(tmp_path / "test.json")
    agg.save_json(path)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 1


# ── Shuffled regime env (Model C) ─────────────────────────────────────────────

def test_shuffled_regime_env_obs_shape():
    """ShuffledRegimeEnv must produce 12-dim observations."""
    from src.evaluation.ablation import ShuffledRegimeEnv
    df = generate_scenario_data("normal", n_steps=50, seed=0)
    env = ShuffledRegimeEnv(market_data=df, horizon_steps=20)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,), f"Expected (12,), got {obs.shape}"


def test_shuffled_regime_probs_sum_to_one():
    """Shuffled regime probabilities [obs[8:12]] must sum to 1."""
    from src.evaluation.ablation import ShuffledRegimeEnv
    df = generate_scenario_data("normal", n_steps=50, seed=0)
    env = ShuffledRegimeEnv(market_data=df, horizon_steps=20)
    obs, _ = env.reset(seed=0)
    prob_sum = float(obs[8:12].sum())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probs sum to {prob_sum}"


def test_shuffled_regime_rollout_completes():
    """Full rollout of ShuffledRegimeEnv must complete without errors."""
    from src.evaluation.ablation import ShuffledRegimeEnv
    df = generate_scenario_data("normal", n_steps=50, seed=0)
    env = ShuffledRegimeEnv(market_data=df, horizon_steps=20, target_inventory=5_000.0)
    obs, _ = env.reset(seed=5)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    assert info["shuffled_regime"] is True


# ── Experiment runner smoke test ───────────────────────────────────────────────

def test_experiment_runner_smoke(tmp_path):
    """
    Smoke test: run 1 scenario × 1 seed × 500 timesteps.
    Verifies runner completes and output files are created.
    """
    from src.evaluation.experiment_runner import run_experiment_suite

    agg = run_experiment_suite(
        scenarios=["normal"],
        seeds=[42],
        train_timesteps=500,
        results_dir=str(tmp_path),
        verbose=0,
        run_ablation=True,
    )

    df = agg.to_dataframe()
    assert len(df) > 0, "No records collected"

    # Check output files exist
    for fname in [
        "experiment_results.csv",
        "summary_table.csv",
        "comparison_table.csv",
        "rq_summary.json",
    ]:
        assert os.path.exists(str(tmp_path / fname)), f"Missing output: {fname}"

    # Every record must have a strategy name and valid completion rate
    assert df["completion_rate"].between(0.0, 1.0 + 1e-9).all()
    assert df["strategy_name"].notna().all()
