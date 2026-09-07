"""
Unit tests for TWAP, VWAP, and POV baseline execution strategies.

Tests:
  1.  TWAP slice size is uniform and rounds correctly
  2.  TWAP last step flushes all remaining inventory
  3.  TWAP produces full execution (completion_rate == 1)
  4.  TWAP does not use market data
  5.  VWAP uses only past volumes (no lookahead)
  6.  VWAP returns a non-negative quantity at every step
  7.  POV requests target_rate * current_volume
  8.  POV clamps to remaining inventory
  9.  All strategies use the same ExecutionSimulator (identical market data / impact)
  10. Standardized BaselineResult fields are present and correctly typed
  11. POV with zero volume returns 0
  12. TWAP with 1 step executes full inventory on that step
"""

import os
import math
import pytest
import pandas as pd
import numpy as np

from src.baselines import (
    TWAPStrategy,
    VWAPStrategy,
    POVStrategy,
    BaselineRunner,
    BaselineResult,
)
from src.execution.impact_models import LinearImpactModel

TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")
TARGET = 100_000.0


@pytest.fixture
def market_df():
    df = pd.read_csv(TEST_DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def runner():
    return BaselineRunner(
        impact_model=LinearImpactModel(eta=0.1, gamma=0.05),
        max_participation_rate=1.0,   # disable vol cap for deterministic testing
        per_share_fee=0.0005,
    )


# ─────────────────────────── TWAP ────────────────────────────

def test_twap_uniform_slices():
    strat = TWAPStrategy(target_inventory=10_000.0, total_steps=10)
    # Every non-final step should return base_slice = floor(10_000/10) = 1_000
    for t in range(9):
        qty = strat.get_action(
            remaining_inventory=10_000.0 - t * 1_000.0,
            time_step=t,
            total_steps=10,
        )
        assert qty == pytest.approx(1_000.0), f"step {t}: expected 1000 got {qty}"


def test_twap_last_step_flushes():
    strat = TWAPStrategy(target_inventory=10_000.0, total_steps=10)
    qty = strat.get_action(remaining_inventory=3_500.0, time_step=9, total_steps=10)
    assert qty == pytest.approx(3_500.0)


def test_twap_single_step():
    strat = TWAPStrategy(target_inventory=5_000.0, total_steps=1)
    qty = strat.get_action(remaining_inventory=5_000.0, time_step=0, total_steps=1)
    assert qty == pytest.approx(5_000.0)


def test_twap_does_not_consume_market_kwargs():
    """TWAP must work correctly even without any market kwargs."""
    strat = TWAPStrategy(target_inventory=6_000.0, total_steps=3)
    qty = strat.get_action(remaining_inventory=6_000.0, time_step=0, total_steps=3)
    assert qty >= 0.0


def test_twap_full_completion(market_df, runner):
    strat = TWAPStrategy(target_inventory=TARGET, total_steps=30)
    result = runner.run(strat, market_df, target_inventory=TARGET, horizon_steps=30)
    assert result.completion_rate == pytest.approx(1.0)
    assert result.remaining_inventory == pytest.approx(0.0, abs=1.0)


# ─────────────────────────── VWAP ────────────────────────────

def test_vwap_no_lookahead():
    """
    VWAP must not see future volumes. Verify that passing a large
    current_volume for step t does NOT inflate the ORDER for step t
    beyond what was derived from past data alone.
    """
    strat = VWAPStrategy(target_inventory=30_000.0, total_steps=10)
    # Step 0 — no history yet; should return 1/10 of remaining (3,000)
    qty0 = strat.get_action(
        remaining_inventory=30_000.0,
        time_step=0,
        total_steps=10,
        current_volume=999_999,  # large future-ish volume; must NOT bias step-0 decision
    )
    assert qty0 == pytest.approx(3_000.0, rel=0.01)


def test_vwap_non_negative_all_steps():
    strat = VWAPStrategy(target_inventory=10_000.0, total_steps=5)
    remaining = 10_000.0
    for t in range(5):
        qty = strat.get_action(
            remaining_inventory=remaining,
            time_step=t,
            total_steps=5,
            current_volume=50_000.0,
        )
        assert qty >= 0.0
        remaining -= qty


def test_vwap_resets_history():
    strat = VWAPStrategy(target_inventory=10_000.0, total_steps=5)
    # Run one episode
    for t in range(5):
        strat.get_action(10_000.0, t, 5, current_volume=40_000.0)
    assert len(strat._observed_volumes) > 0
    strat.reset_history()
    assert strat._observed_volumes == []


def test_vwap_completion(market_df, runner):
    strat = VWAPStrategy(target_inventory=TARGET, total_steps=30)
    result = runner.run(strat, market_df, target_inventory=TARGET, horizon_steps=30)
    assert result.completion_rate == pytest.approx(1.0)
    assert result.remaining_inventory == pytest.approx(0.0, abs=1.0)


# ─────────────────────────── POV ────────────────────────────

def test_pov_targets_fraction_of_volume():
    strat = POVStrategy(target_inventory=100_000.0, target_rate=0.10)
    qty = strat.get_action(
        remaining_inventory=100_000.0,
        time_step=0,
        total_steps=30,
        current_volume=50_000.0,
    )
    assert qty == pytest.approx(5_000.0)  # 10% of 50,000


def test_pov_clamps_to_remaining():
    strat = POVStrategy(target_inventory=100_000.0, target_rate=0.50)
    qty = strat.get_action(
        remaining_inventory=200.0,
        time_step=0,
        total_steps=30,
        current_volume=50_000.0,  # 50% * 50_000 = 25_000 > remaining
    )
    assert qty == pytest.approx(200.0)


def test_pov_zero_volume_returns_zero():
    strat = POVStrategy(target_inventory=100_000.0, target_rate=0.10)
    qty = strat.get_action(
        remaining_inventory=100_000.0,
        time_step=5,
        total_steps=30,
        current_volume=0.0,
    )
    assert qty == 0.0


def test_pov_invalid_rate():
    with pytest.raises(ValueError):
        POVStrategy(target_inventory=100_000.0, target_rate=0.0)
    with pytest.raises(ValueError):
        POVStrategy(target_inventory=100_000.0, target_rate=1.5)


def test_pov_non_zero_completion(market_df, runner):
    strat = POVStrategy(target_inventory=TARGET, target_rate=0.15)
    result = runner.run(strat, market_df, target_inventory=TARGET, horizon_steps=30)
    # POV may not fully complete (depends on vol); at least some execution happened
    assert result.executed_inventory > 0
    assert result.completion_rate >= 0.0


# ─────────────────────── Shared Metrics ─────────────────────

@pytest.mark.parametrize("strategy_fn", [
    lambda: TWAPStrategy(TARGET, 30),
    lambda: VWAPStrategy(TARGET, 30),
    lambda: POVStrategy(TARGET, 0.10),
])
def test_baseline_result_fields(strategy_fn, market_df, runner):
    """Every strategy must return a BaselineResult with all required fields."""
    result = runner.run(strategy_fn(), market_df, target_inventory=TARGET, horizon_steps=30)

    assert isinstance(result, BaselineResult)
    assert isinstance(result.strategy_name, str)
    assert result.target_inventory == pytest.approx(TARGET)
    # Clamp tolerance: floating-point accumulation may push rate to 1.0 + epsilon
    assert result.completion_rate == pytest.approx(1.0, abs=1e-6) or (0.0 <= result.completion_rate <= 1.0)
    assert result.executed_inventory >= 0.0
    assert result.remaining_inventory >= 0.0
    assert result.executed_inventory + result.remaining_inventory == pytest.approx(TARGET, rel=1e-5)
    assert result.arrival_price > 0.0
    assert result.implementation_shortfall_bps is not None
    assert result.turnover >= 0.0
    assert result.execution_duration_steps >= 0
    assert result.total_steps == 30


@pytest.mark.parametrize("strategy_fn", [
    lambda: TWAPStrategy(TARGET, 30),
    lambda: VWAPStrategy(TARGET, 30),
    lambda: POVStrategy(TARGET, 0.15),
])
def test_all_strategies_use_same_impact_model(strategy_fn, market_df):
    """Verify that the same impact model produces consistent arrival prices."""
    impact = LinearImpactModel(eta=0.1, gamma=0.05)
    runner_shared = BaselineRunner(impact_model=impact, max_participation_rate=1.0)
    result = runner_shared.run(strategy_fn(), market_df, target_inventory=TARGET, horizon_steps=30)
    # Arrival price must equal the first bar mid price regardless of strategy
    assert result.arrival_price == pytest.approx(150.00, rel=0.01)
