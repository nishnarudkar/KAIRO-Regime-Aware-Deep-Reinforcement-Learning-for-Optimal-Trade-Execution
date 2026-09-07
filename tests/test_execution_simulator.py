"""
Unit tests for ExecutionSimulator engine.

Tests:
1. Inventory accounting (remaining_inventory + executed_inventory == target_inventory)
2. Order completion (simulator reaches done state when target fulfilled)
3. No negative inventory (attempting to execute > remaining inventory is clamped)
4. Time advancement (steps advance sequentially, elapsed & remaining time updated correctly)
5. Execution cost (spread, fees, and execution price formulas)
6. Market impact (temporary impact inflates exec price, permanent impact shifts future mid prices)
7. Terminal conditions (unexecuted inventory penalty at horizon end)
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.execution import (
    ExecutionSimulator,
    LinearImpactModel,
    AlmgrenChrissImpactModel
)

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")


@pytest.fixture
def test_market_df():
    """Load deterministic test dataset."""
    df = pd.read_csv(TEST_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def test_inventory_accounting(test_market_df):
    """Verify remaining_inventory + executed_inventory always equals target_inventory."""
    sim = ExecutionSimulator()
    target_qty = 100000.0
    sim.reset(test_market_df, target_inventory=target_qty, side="BUY")

    assert sim.remaining_inventory == target_qty
    assert sim.executed_inventory == 0.0

    # Step 1: execute 30,000 shares
    res1 = sim.step(30000.0)
    assert sim.remaining_inventory + sim.executed_inventory == pytest.approx(target_qty)
    assert sim.executed_inventory == res1.filled_qty

    # Step 2: execute 40,000 shares
    res2 = sim.step(40000.0)
    assert sim.remaining_inventory + sim.executed_inventory == pytest.approx(target_qty)


def test_no_negative_inventory(test_market_df):
    """Verify executing more than remaining inventory is clamped and remaining inventory never goes negative."""
    sim = ExecutionSimulator(max_participation_rate=1.0)  # no volume cap for this test
    target_qty = 10000.0
    sim.reset(test_market_df, target_inventory=target_qty, side="BUY")

    # Request 50,000 shares when target is 10,000
    res = sim.step(50000.0)

    assert res.filled_qty == pytest.approx(10000.0)
    assert sim.remaining_inventory == 0.0
    assert sim.executed_inventory == 10000.0
    assert sim.is_done() is True


def test_order_completion(test_market_df):
    """Verify that simulator marks simulation as done when order is fully executed."""
    sim = ExecutionSimulator(max_participation_rate=1.0)
    target_qty = 20000.0
    sim.reset(test_market_df, target_inventory=target_qty, side="BUY")

    sim.step(10000.0)
    assert sim.is_done() is False

    sim.step(10000.0)
    assert sim.remaining_inventory == 0.0
    assert sim.is_done() is True


def test_time_advancement(test_market_df):
    """Verify steps advance sequentially and remaining/elapsed time update correctly."""
    sim = ExecutionSimulator()
    sim.reset(test_market_df, target_inventory=100000.0, side="BUY", horizon_steps=30)

    state0 = sim.get_state()
    assert state0["current_step"] == 0
    assert state0["elapsed_time"] == 0
    assert state0["remaining_time"] == 30

    sim.step(1000.0)
    state1 = sim.get_state()
    assert state1["current_step"] == 1
    assert state1["elapsed_time"] == 1
    assert state1["remaining_time"] == 29
    assert state1["current_timestamp"] == test_market_df.loc[1, 'timestamp']


def test_execution_cost_and_fees(test_market_df):
    """Verify spread cost, transaction fees, and effective execution price calculations."""
    per_share_fee = 0.001
    per_trade_fee = 1.0
    bps_fee = 1.0  # 1 bps

    sim = ExecutionSimulator(
        per_share_fee=per_share_fee,
        per_trade_fee=per_trade_fee,
        bps_fee=bps_fee,
        max_participation_rate=1.0
    )
    sim.reset(test_market_df, target_inventory=1000.0, side="BUY")

    # Step at t=0: price=150.00, ask=150.02
    res = sim.step(1000.0)

    # Base ask is 150.02
    expected_fees = 1.0 + (0.001 * 1000.0) + (1.0 * 0.0001 * (1000.0 * res.execution_price))
    assert res.transaction_fee == pytest.approx(expected_fees, rel=1e-4)
    assert res.execution_price > res.ask_price  # temporary impact elevates price above ask for BUY


def test_market_impact(test_market_df):
    """Verify temporary impact elevates execution price and permanent impact shifts future mid prices."""
    impact_model = LinearImpactModel(eta=0.1, gamma=0.05)
    sim = ExecutionSimulator(impact_model=impact_model, max_participation_rate=1.0)
    sim.reset(test_market_df, target_inventory=50000.0, side="BUY")

    row0 = test_market_df.iloc[0]
    # Trade size = 10,000, Bar volume = 50,000 -> participation rate = 0.2
    # temp_impact = 1.0 * 0.1 * 0.2 * 150.00 = 3.00
    # perm_impact = 1.0 * 0.05 * 0.2 * 150.00 = 1.50
    res0 = sim.step(10000.0)

    assert res0.temporary_impact == pytest.approx(3.00)
    assert res0.permanent_impact == pytest.approx(1.50)
    assert res0.execution_price == pytest.approx(row0['ask'] + 3.00)

    # Check that t=1 mid price incorporates accumulated permanent impact (150.10 + 1.50 = 151.60)
    state1 = sim.get_state()
    assert state1['current_price'] == pytest.approx(test_market_df.loc[1, 'price'] + 1.50)


def test_terminal_conditions_and_penalty(test_market_df):
    """Verify terminal penalty is computed when remaining inventory > 0 at horizon end."""
    sim = ExecutionSimulator(max_participation_rate=0.05)  # low volume cap forces incomplete fill
    sim.reset(test_market_df, target_inventory=100000.0, side="BUY", horizon_steps=5)

    # Execute for 5 steps
    for _ in range(5):
        sim.step(20000.0)

    assert sim.is_done() is True
    assert sim.remaining_inventory > 0.0

    metrics = sim.compute_metrics()
    assert metrics.terminal_penalty > 0.0
    assert metrics.implementation_shortfall > 0.0
    assert metrics.remaining_inventory == sim.remaining_inventory
