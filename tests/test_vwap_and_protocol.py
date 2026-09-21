"""
Tests for the VWAP baseline (which used to be TWAP in disguise) and the shared
evaluation protocol.
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import TWAPStrategy, VWAPStrategy
from src.baselines.runner import BaselineRunner
from src.baselines.vwap import estimate_volume_profile
from src.evaluation import protocol as P
from src.evaluation.scenarios import generate_scenario_data


@pytest.fixture(scope="module")
def df():
    return generate_scenario_data("normal", 3000, seed=42)


# ── VWAP ──────────────────────────────────────────────────────────────────────

def test_vwap_without_history_is_twap_like(df):
    window = df.iloc[100:130].reset_index(drop=True)
    runner = BaselineRunner()
    twap = runner.run(TWAPStrategy(100_000, 30), window, 100_000, "BUY", 30)
    vwap = runner.run(VWAPStrategy(100_000, 30), window, 100_000, "BUY", 30, history=None)
    # A flat profile re-divides the remaining inventory each step (TWAP uses fixed slices),
    # so they agree up to partial-fill effects.
    assert vwap.implementation_shortfall_bps == pytest.approx(twap.implementation_shortfall_bps, abs=0.05)


def test_vwap_with_history_follows_the_volume_profile(df):
    start = 2100
    window = df.iloc[start:start + 30].reset_index(drop=True)
    history = df.iloc[:start]
    strat = VWAPStrategy(100_000, 30)
    runner = BaselineRunner()
    runner.run(strat, window, 100_000, "BUY", 30, history=history)
    assert strat.uses_profile

    slices = np.array([s.requested_qty for s in runner.simulator.execution_history])
    twap_slices = np.array([100_000 / 30] * 30)
    assert not np.allclose(slices[:-1], twap_slices[:-1], rtol=0.02)   # differs from flat TWAP
    assert slices.sum() == pytest.approx(100_000, rel=1e-6)


def test_vwap_profile_uses_only_history(df):
    start = 2100
    window = df.iloc[start:start + 30]
    p1 = estimate_volume_profile(df.iloc[:start], window)
    tampered = df.copy()
    tampered.loc[start:, "volume"] *= 10.0        # current + future bars
    p2 = estimate_volume_profile(tampered.iloc[:start], window)
    np.testing.assert_allclose(p1, p2)


def test_vwap_profile_none_without_enough_history(df):
    assert estimate_volume_profile(df.iloc[:10], df.iloc[100:130]) is None


# ── Protocol ──────────────────────────────────────────────────────────────────

def test_chronological_split_and_windows(df):
    sp = P.make_split(df)
    assert sp.cut == int(len(df) * 0.7)
    assert sp.train["timestamp"].max() < sp.full["timestamp"].iloc[sp.cut]
    starts = P.test_window_starts(sp)
    assert starts[0] == sp.cut
    assert all(b - a == P.HORIZON_STEPS for a, b in zip(starts, starts[1:]))     # non-overlapping
    assert starts[-1] + P.HORIZON_STEPS <= sp.n
    lo, hi = P.train_start_range(sp.cut)
    assert hi + P.HORIZON_STEPS <= sp.cut                                        # train windows never touch test


def test_baselines_are_evaluated_on_identical_windows(df):
    sp = P.make_split(df)
    starts = P.test_window_starts(sp, max_windows=4)
    res = P.evaluate_baseline_windows(sp.full, starts)
    assert set(res["strategy"]) == set(P.BASELINE_NAMES)
    for name in P.BASELINE_NAMES:
        assert list(res[res.strategy == name]["window_start"]) == starts
    # arrival price is a property of the window, identical across strategies
    arr = res.pivot(index="window", columns="strategy", values="arrival_price")
    assert np.allclose(arr["TWAP"], arr["POV"])


def test_hmm_is_fitted_on_train_only(df):
    sp = P.make_split(df)
    hmm = P.fit_hmm([sp.train])
    assert hmm is not None and hmm.is_fitted and hmm.n_regimes == 4


def test_paired_stats_detects_a_shift_and_reports_noise():
    rng = np.random.default_rng(0)
    shifted = rng.normal(-3.0, 1.0, 60)
    s = P.paired_stats(shifted)
    assert s["mean"] < -2.0 and s["ci_high"] < 0 and s["p_value"] < 0.001 and s["win_rate"] > 0.9

    noise = P.paired_stats(rng.normal(0.0, 5.0, 30))
    assert noise["ci_low"] < 0 < noise["ci_high"]


def test_paired_stats_handles_empty_and_nan():
    assert P.paired_stats([])["n"] == 0
    assert P.paired_stats([np.nan, 1.0, 2.0])["n"] == 2


def test_agent_window_rollout_records_trajectories(df):
    sp = P.make_split(df)
    hmm = P.fit_hmm([sp.train])
    agent = P.train_agent("dqn", "regime", [sp.train], hmm, seed=0, timesteps=300, cut=sp.cut)
    env = P.build_env("regime", [sp.full], hmm)
    out = P.rollout_agent_window(agent, env, sp.cut, record_obs=True)
    assert len(out["inventory_trajectory"]) == P.HORIZON_STEPS + 1 or out["inventory_trajectory"][-1] == 0
    assert len(out["observation_trajectory"][0]) == 12
    assert len(out["action_trajectory"]) == len(out["observation_trajectory"])
    assert out["window_start"] == sp.cut


def test_experiment_suite_smoke(tmp_path):
    """End-to-end: paired windows, statistics and every output file, with tiny training."""
    from src.evaluation.experiment_runner import run_experiment_suite

    agg = run_experiment_suite(
        scenarios=["normal"], seeds=[1, 2], train_timesteps=300, results_dir=str(tmp_path),
        include_ppo=False, run_ablation=True, n_bars=900, max_windows=3, n_jobs=1,
    )
    df = agg.to_dataframe()
    assert set(df["strategy_name"]) >= {"TWAP", "VWAP", "POV", "DQN", "Regime-Aware DQN", "DQN (shuffled regime)"}
    for fname in ["window_results.csv", "experiment_results.csv", "summary_table.csv", "comparison_table.csv",
                  "per_scenario_table.csv", "paired_comparisons.csv", "hmm_validation.csv",
                  "rq_summary.json", "run_config.json"]:
        assert (tmp_path / fname).exists(), fname
    win = pd.read_csv(tmp_path / "window_results.csv")
    assert win.groupby(["scenario", "seed", "strategy"]).size().nunique() == 1      # same windows everywhere
    comp = pd.read_csv(tmp_path / "paired_comparisons.csv")
    assert {"window", "seed"} <= set(comp["level"])
    assert {"RQ1", "RQ2", "RQ3-control"} <= set(comp["rq"])
