"""
Unit tests for Gymnasium TradeExecutionEnv environment.

Tests:
1. Gymnasium environment API validation via check_env
2. Deterministic seeding (reproducibility across reset and step rollouts)
3. Observation space shape & bounds
4. Action space execution & inventory decay
5. Reward computation validity & components
6. Episode termination & truncation conditions
"""

import os
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.environment import TradeExecutionEnv, ModularExecutionReward

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")


@pytest.fixture
def test_market_df():
    """Load deterministic 30-min market test dataset."""
    df = pd.read_csv(TEST_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def test_gymnasium_env_validation(test_market_df):
    """Validate Gymnasium API compliance using gymnasium.utils.env_checker.check_env."""
    env = TradeExecutionEnv(market_data=test_market_df, target_inventory=100000.0, horizon_steps=30)
    
    # check_env checks spaces, reset return signature, step return signature, observation bounds
    check_env(env.unwrapped, skip_render_check=True)


def test_observation_and_action_spaces(test_market_df):
    """Verify observation space is Box(7,) and action space is Discrete(4)."""
    env = TradeExecutionEnv(market_data=test_market_df, target_inventory=50000.0, horizon_steps=30)
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (7,)
    assert env.observation_space.dtype == np.float32

    obs, info = env.reset()
    assert obs.shape == (7,)
    assert env.observation_space.contains(obs)


def test_deterministic_seeding(test_market_df):
    """Verify deterministic reset reproducibility using seeds."""
    env1 = TradeExecutionEnv(market_data=test_market_df, target_inventory=100000.0)
    env2 = TradeExecutionEnv(market_data=test_market_df, target_inventory=100000.0)

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1, obs2)
    assert info1["remaining_inventory"] == info2["remaining_inventory"]

    # Rollout identical sequence of actions
    actions = [1, 2, 0, 3, 1]
    for act in actions:
        o1, r1, term1, trunc1, i1 = env1.step(act)
        o2, r2, term2, trunc2, i2 = env2.step(act)

        np.testing.assert_array_almost_equal(o1, o2)
        assert r1 == pytest.approx(r2)
        assert term1 == term2


def test_action_execution_and_inventory_decay(test_market_df):
    """Verify discrete actions execute inventory fractions correctly."""
    env = TradeExecutionEnv(
        market_data=test_market_df,
        target_inventory=100000.0,
        horizon_steps=30,
        max_participation_rate=1.0  # Allow full fill without bar volume cap
    )
    obs, info = env.reset()

    assert info["remaining_inventory"] == 100000.0

    # Action 2 = execute 25% of remaining (25,000 shares)
    obs, reward, terminated, truncated, info = env.step(2)
    assert info["remaining_inventory"] == pytest.approx(75000.0)
    assert info["filled_qty"] == pytest.approx(25000.0)

    # Action 0 = hold (0% executed)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["remaining_inventory"] == pytest.approx(75000.0)
    assert info["filled_qty"] == 0.0


def test_reward_computation(test_market_df):
    """Verify reward is calculated properly with non-positive penalty values."""
    reward_calc = ModularExecutionReward(lambda_cost=1.0, lambda_impact=1.0, lambda_risk=1e-4)
    env = TradeExecutionEnv(
        market_data=test_market_df,
        target_inventory=100000.0,
        reward_calculator=reward_calc
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)  # Action 3 = execute 50%

    assert isinstance(reward, float)
    assert reward <= 0.0  # Reward is negative penalty
    assert "reward_cost_penalty" in info
    assert "reward_impact_penalty" in info


def test_random_action_rollout_and_termination(test_market_df):
    """Verify episode terminates properly when inventory is fully executed or horizon reached."""
    env = TradeExecutionEnv(market_data=test_market_df, target_inventory=20000.0, horizon_steps=10)
    obs, info = env.reset(seed=123)

    step_count = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

    assert done is True
    assert step_count <= 10
