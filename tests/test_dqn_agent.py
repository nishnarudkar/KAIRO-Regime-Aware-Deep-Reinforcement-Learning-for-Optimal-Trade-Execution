"""
tests/test_dqn_agent.py — Unit tests for Stage 6 DQN Agent

Tests follow the project hard-rules:
  - All tests use ONLY synthetic deterministic data (no live API calls).
  - Seeds are fixed for reproducibility.
  - No look-ahead: the environment receives data row-by-row.
  - DQN agent is tested with minimal timesteps (smoke tests, not research runs).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure repo root is on path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_market_data(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic market data (no real API calls)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")
    prices = 150.0 + np.cumsum(rng.normal(0, 0.05, n))
    volumes = rng.integers(30_000, 80_000, n).astype(float)
    spreads = rng.uniform(0.02, 0.06, n)
    vols = rng.uniform(0.001, 0.003, n)

    return pd.DataFrame({
        "timestamp": dates,
        "price":     prices,
        "volume":    volumes,
        "spread":    spreads,
        "volatility": vols,
    })


@pytest.fixture
def market_data():
    return make_market_data(n=40, seed=42)


@pytest.fixture
def env(market_data):
    """Return a TradeExecutionEnv configured with synthetic data."""
    from src.environment import TradeExecutionEnv
    return TradeExecutionEnv(
        market_data=market_data,
        target_inventory=10_000.0,
        side="BUY",
        horizon_steps=30,
    )


@pytest.fixture
def trained_agent(env):
    """Minimal DQN agent (1 000 steps) — just enough for a smoke test."""
    from src.agents.dqn_agent import DQNAgent
    agent = DQNAgent(env=env, seed=42, verbose=0, learning_starts=100)
    agent.train(total_timesteps=1_000)
    return agent


# ── Basic Instantiation ────────────────────────────────────────────────────────

def test_dqn_agent_instantiates(env):
    """DQNAgent must be constructable without errors."""
    from src.agents.dqn_agent import DQNAgent
    agent = DQNAgent(env=env, seed=0, verbose=0)
    assert agent is not None


def test_dqn_agent_predict_before_train(env):
    """Untrained DQN should still return a valid action (random policy)."""
    from src.agents.dqn_agent import DQNAgent
    agent = DQNAgent(env=env, seed=0, verbose=0)
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=False)
    assert action in {0, 1, 2, 3}, f"Expected action in [0,1,2,3], got {action}"


# ── Training Smoke Test ────────────────────────────────────────────────────────

def test_dqn_trains_without_error(env):
    """Training for a small number of steps must complete without raising."""
    from src.agents.dqn_agent import DQNAgent
    agent = DQNAgent(env=env, seed=42, verbose=0, learning_starts=50)
    agent.train(total_timesteps=500)   # smoke: fast, not research-quality


def test_dqn_predict_after_train(trained_agent, env):
    """After training, predict must return valid discrete action."""
    obs, _ = env.reset(seed=1)
    action = trained_agent.predict(obs, deterministic=True)
    assert action in {0, 1, 2, 3}


# ── Action Semantics ────────────────────────────────────────────────────────────

def test_dqn_action_space_valid(trained_agent, env):
    """Agent must only produce actions within the environment's action space."""
    obs, _ = env.reset(seed=2)
    for _ in range(10):
        action = trained_agent.predict(obs, deterministic=True)
        assert env.action_space.contains(action), f"Invalid action: {action}"
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


# ── Save / Load Round-trip ────────────────────────────────────────────────────

def test_dqn_save_and_load(trained_agent, env, tmp_path):
    """
    Save the trained model then load it.
    The loaded model must produce identical predictions for a fixed observation.
    """
    save_path = str(tmp_path / "test_dqn")
    trained_agent.save(save_path)
    assert os.path.exists(save_path + ".zip"), "Saved checkpoint file not found."

    from src.agents.dqn_agent import DQNAgent
    loaded_agent = DQNAgent.load(save_path, env=env)

    obs, _ = env.reset(seed=3)
    action_original = trained_agent.predict(obs, deterministic=True)
    action_loaded   = loaded_agent.predict(obs,   deterministic=True)
    assert action_original == action_loaded, (
        f"Action mismatch after load: original={action_original}, loaded={action_loaded}"
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_evaluate_agent_returns_result(trained_agent, market_data):
    """evaluate_agent must return an AgentEvalResult with valid metric ranges."""
    from src.agents.evaluator import evaluate_agent, AgentEvalResult

    result = evaluate_agent(
        agent=trained_agent,
        market_data=market_data,
        target_inventory=10_000.0,
        side="BUY",
        horizon_steps=30,
        seed=42,
        agent_name="DQN_test",
    )

    assert isinstance(result, AgentEvalResult)
    assert result.completion_rate >= 0.0
    assert result.completion_rate <= 1.0 + 1e-9
    assert result.total_steps == 30
    assert result.target_inventory == pytest.approx(10_000.0, rel=1e-6)
    assert len(result.inventory_trajectory) >= 1
    assert len(result.action_trajectory) >= 1
    assert all(a in {0, 1, 2, 3} for a in result.action_trajectory)


def test_evaluate_agent_action_distribution(trained_agent, market_data):
    """Action counts must sum to total number of steps taken."""
    from src.agents.evaluator import evaluate_agent

    result = evaluate_agent(
        agent=trained_agent,
        market_data=market_data,
        target_inventory=10_000.0,
        horizon_steps=30,
        seed=42,
    )
    total_actions = sum(result.action_counts.values())
    assert total_actions == len(result.action_trajectory), (
        "action_counts total must equal length of action_trajectory"
    )


def test_evaluate_agent_inventory_non_negative(trained_agent, market_data):
    """Remaining inventory must never go negative."""
    from src.agents.evaluator import evaluate_agent

    result = evaluate_agent(
        agent=trained_agent,
        market_data=market_data,
        target_inventory=10_000.0,
        horizon_steps=30,
        seed=42,
    )
    for inv in result.inventory_trajectory:
        assert inv >= -1e-6, f"Negative inventory detected: {inv}"


# ── Determinism ────────────────────────────────────────────────────────────────

def test_dqn_deterministic_predict(trained_agent, env):
    """
    Two calls to predict with deterministic=True on the same obs
    must return the same action.
    """
    obs, _ = env.reset(seed=7)
    a1 = trained_agent.predict(obs, deterministic=True)
    a2 = trained_agent.predict(obs, deterministic=True)
    assert a1 == a2, "Deterministic predict must be reproducible"
