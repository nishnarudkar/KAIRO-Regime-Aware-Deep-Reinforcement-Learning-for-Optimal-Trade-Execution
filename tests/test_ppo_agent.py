"""
tests/test_ppo_agent.py — Unit tests for Stage 9 PPO Agent & Experiment Suite

Tests cover:
  - PPOAgent initialization with default/custom hyperparameters
  - Prediction before & after training (discrete action space in {0, 1, 2, 3})
  - Training on base TradeExecutionEnv (7-dim)
  - Training on RegimeAwareTradeExecutionEnv (12-dim)
  - Model serialization (save and load)
  - run_ppo_experiment multi-algorithm comparison smoke test
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.agents.ppo_agent import PPOAgent
from src.agents.evaluator import evaluate_agent
from src.environment import TradeExecutionEnv, RegimeAwareTradeExecutionEnv
from src.regimes.hmm_model import MarketHMM
from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler


@pytest.fixture
def sample_market_data():
    """Generates synthetic 1-minute OHLCV market data (300 points for robust HMM fitting)."""
    dates = pd.date_range("2025-01-02 09:30", periods=300, freq="1min")
    rng = np.random.default_rng(42)
    close = 150.0 + np.cumsum(rng.normal(0, 0.05, 300))
    high = close + rng.uniform(0.02, 0.15, 300)
    low  = close - rng.uniform(0.02, 0.15, 300)
    volume = rng.integers(30000, 80000, 300).astype(float)
    spread = rng.uniform(0.02, 0.06, 300)
    volatility = rng.uniform(0.001, 0.003, 300)

    return pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "price": close,
        "volume": volume,
        "spread": spread,
        "volatility": volatility,
    })


@pytest.fixture
def base_env(sample_market_data):
    """Instantiates base TradeExecutionEnv (7-dim observation space)."""
    return TradeExecutionEnv(
        market_data=sample_market_data,
        target_inventory=10000.0,
        horizon_steps=30,
    )


@pytest.fixture
def regime_env(sample_market_data):
    """Instantiates RegimeAwareTradeExecutionEnv (12-dim observation space)."""
    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(sample_market_data)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)
    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42)
    hmm.fit(X, scaler=scaler)

    return RegimeAwareTradeExecutionEnv(
        hmm_model=hmm,
        regime_feature_data=feat_df,
        market_data=sample_market_data,
        target_inventory=10000.0,
        horizon_steps=30,
    )


def test_ppo_agent_instantiates(base_env):
    """Verify PPOAgent instantiates with default and custom hyperparams."""
    agent = PPOAgent(env=base_env, seed=42, verbose=0)
    assert agent is not None
    assert agent.env == base_env

    # Custom hyperparams
    agent_custom = PPOAgent(
        env=base_env,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=32,
        seed=123,
        verbose=0,
    )
    assert agent_custom is not None


def test_ppo_agent_predict_before_train(base_env):
    """Verify predict returns a valid discrete action in {0, 1, 2, 3} before training."""
    agent = PPOAgent(env=base_env, seed=42, verbose=0)
    state, _ = base_env.reset()
    action = agent.predict(state, deterministic=True)

    assert isinstance(action, (int, np.integer))
    assert action in {0, 1, 2, 3}


def test_ppo_trains_base_env(base_env):
    """Verify PPO trains on 7-dim base env without error."""
    agent = PPOAgent(env=base_env, seed=42, n_steps=256, batch_size=32, verbose=0)
    agent.train(total_timesteps=512)
    assert agent.model.num_timesteps >= 512


def test_ppo_trains_regime_env(regime_env):
    """Verify PPO trains on 12-dim regime-aware env without error."""
    agent = PPOAgent(env=regime_env, seed=42, n_steps=256, batch_size=32, verbose=0)
    agent.train(total_timesteps=512)
    assert agent.model.num_timesteps >= 512


def test_ppo_predict_after_train(base_env):
    """Verify prediction after training is deterministic when requested."""
    agent = PPOAgent(env=base_env, seed=42, n_steps=256, batch_size=32, verbose=0)
    agent.train(total_timesteps=512)
    state, _ = base_env.reset()

    a1 = agent.predict(state, deterministic=True)
    a2 = agent.predict(state, deterministic=True)
    assert a1 == a2
    assert a1 in {0, 1, 2, 3}


def test_ppo_save_and_load(base_env):
    """Verify PPO model can be saved and loaded from disk."""
    agent = PPOAgent(env=base_env, seed=42, n_steps=256, batch_size=32, verbose=0)
    agent.train(total_timesteps=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "ppo_model")
        agent.save(save_path)

        # SB3 adds .zip automatically
        zip_path = save_path + ".zip" if not save_path.endswith(".zip") else save_path
        assert os.path.exists(zip_path)

        loaded_agent = PPOAgent.load(save_path, env=base_env)
        assert loaded_agent is not None

        state, _ = base_env.reset()
        a_orig = agent.predict(state, deterministic=True)
        a_load = loaded_agent.predict(state, deterministic=True)
        assert a_orig == a_load


def test_ppo_evaluator_integration(base_env, sample_market_data):
    """Verify evaluate_agent works cleanly with a PPO agent."""
    agent = PPOAgent(env=base_env, seed=42, n_steps=256, batch_size=32, verbose=0)
    agent.train(total_timesteps=512)

    result = evaluate_agent(
        agent=agent,
        market_data=sample_market_data,
        target_inventory=10000.0,
        side="BUY",
        horizon_steps=30,
        seed=42,
        agent_name="PPO Test Agent",
    )
    assert result.completion_rate >= 0.0
    assert result.completion_rate <= 1.0
    assert result.execution_cost >= 0.0
    assert isinstance(result.action_counts, dict)
    assert len(result.action_counts) > 0


def test_ppo_experiment_smoke():
    """Smoke test for multi-algorithm run_ppo_experiment runner."""
    from src.agents.ppo_experiment import run_ppo_experiment

    res = run_ppo_experiment(
        train_timesteps=512,
        seed=42,
        verbose=0,
    )
    assert "results" in res
    assert "run_ids" in res
    assert set(res["results"].keys()) == {"A", "B", "C", "D"}
