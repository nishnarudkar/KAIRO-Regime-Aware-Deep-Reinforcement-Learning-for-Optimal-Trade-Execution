"""
Tests for the regime-switching synthetic market generator.

These guard the properties that make experiment results meaningful: realistic
per-minute scales, zero drift, working bid/ask columns, a causal volatility
column, and regimes that actually change the market.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.scenarios import SCENARIOS, REGIME_SIGMA, generate_scenario_data


@pytest.fixture(scope="module")
def normal_df():
    return generate_scenario_data("normal", n_steps=3000, seed=7)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_per_minute_volatility_is_realistic(name):
    df = generate_scenario_data(name, n_steps=2000, seed=1)
    rets = np.log(df["price"]).diff().dropna()
    # The old generator used 5% per minute. Real one-minute volatility is well below 1%.
    assert rets.std() < 0.005
    assert rets.std() > 1e-4


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_prices_stay_in_a_sane_range(name):
    df = generate_scenario_data(name, n_steps=3000, seed=3)
    assert df["price"].min() > 100.0
    assert df["price"].max() < 250.0


def test_zero_drift_across_seeds():
    ends = [np.log(generate_scenario_data("normal", 3000, s)["price"].iloc[-1] / 150.0) for s in range(20)]
    # Mean terminal log-return is indistinguishable from 0 (standard error ~ 0.004)
    assert abs(np.mean(ends)) < 0.02


def test_bid_ask_columns_bracket_price_and_match_spread(normal_df):
    assert {"bid", "ask", "spread"} <= set(normal_df.columns)
    assert (normal_df["bid"] < normal_df["price"]).all()
    assert (normal_df["ask"] > normal_df["price"]).all()
    np.testing.assert_allclose(normal_df["ask"] - normal_df["bid"], normal_df["spread"], rtol=1e-9)


def test_simulator_uses_the_spread_in_the_data():
    """The old data had no bid/ask, so the simulator silently used a constant 2 bps spread."""
    from src.execution import ExecutionSimulator

    df = generate_scenario_data("stress", 200, seed=5)
    sim = ExecutionSimulator()
    state = sim.reset(df, target_inventory=1000, side="BUY", horizon_steps=10)
    assert state["spread"] == pytest.approx(df["spread"].iloc[0], rel=1e-9)


def test_regimes_change_the_market(normal_df):
    by = normal_df.groupby("true_regime")
    sigma = normal_df.assign(r=np.log(normal_df["price"]).diff()).groupby("true_regime")["r"].std()
    assert set(by.groups) >= {0, 1}
    # Realized volatility is ordered like the regime definitions
    ordered = sigma.sort_index().to_numpy()
    assert np.all(np.diff(ordered) > 0)
    assert by["spread"].mean().is_monotonic_increasing


def test_volatility_column_is_causal():
    """Changing only future returns must not change volatility at earlier bars."""
    df = generate_scenario_data("normal", 300, seed=11)
    r = np.log(df["price"]).diff().fillna(0.0).to_numpy()
    from src.evaluation.scenarios import _causal_rolling_vol

    base = _causal_rolling_vol(r, fallback=float(REGIME_SIGMA[1]))
    r2 = r.copy()
    r2[200:] *= 5.0
    changed = _causal_rolling_vol(r2, fallback=float(REGIME_SIGMA[1]))
    np.testing.assert_allclose(base[:200], changed[:200])


def test_timestamps_follow_trading_days():
    df = generate_scenario_data("normal", 800, seed=2)
    ts = pd.to_datetime(df["timestamp"])
    assert ts.iloc[0].strftime("%H:%M") == "09:30"
    assert ts.iloc[389].strftime("%H:%M") == "15:59"
    assert ts.iloc[390].strftime("%H:%M") == "09:30"          # next trading day
    assert ts.iloc[390].date() > ts.iloc[389].date()
    assert ts.is_monotonic_increasing


def test_regime_transition_switches_to_turbulence():
    df = generate_scenario_data("regime_transition", 3000, seed=4)
    early, late = df.iloc[:1500], df.iloc[2000:]
    assert late["true_regime"].mean() > early["true_regime"].mean() + 1.0


def test_liquidity_shock_collapses_volume():
    df = generate_scenario_data("liquidity_shock", 3000, seed=4)
    assert df["volume"].iloc[2000:].mean() < 0.25 * df["volume"].iloc[:1500].mean()
    assert df["spread"].iloc[2000:].mean() > 2.0 * df["spread"].iloc[:1500].mean()


def test_deterministic_and_seed_sensitive():
    a = generate_scenario_data("stress", 500, seed=9)
    b = generate_scenario_data("stress", 500, seed=9)
    c = generate_scenario_data("stress", 500, seed=10)
    pd.testing.assert_frame_equal(a, b)
    assert not a["price"].equals(c["price"])
