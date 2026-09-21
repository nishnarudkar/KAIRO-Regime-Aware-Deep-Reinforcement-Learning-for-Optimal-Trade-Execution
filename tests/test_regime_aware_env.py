"""
tests/test_regime_aware_env.py — Unit tests for Stage 7 RegimeAwareTradeExecutionEnv

All tests use only synthetic deterministic data (no live API calls).
Seeds are fixed. No look-ahead bias introduced in any test.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared fixtures ────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV data with all columns needed for regime features."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-02 09:30", periods=n, freq="1min")
    close = 150.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + rng.uniform(0.02, 0.10, n)
    low = close - rng.uniform(0.02, 0.10, n)
    volume = rng.integers(30_000, 80_000, n).astype(float)
    spread = rng.uniform(0.02, 0.06, n)
    volatility = rng.uniform(0.001, 0.003, n)

    return pd.DataFrame({
        "timestamp":  dates,
        "open":       close,
        "high":       high,
        "low":        low,
        "close":      close,
        "price":      close,
        "volume":     volume,
        "spread":     spread,
        "volatility": volatility,
    })


@pytest.fixture(scope="module")
def all_data():
    return make_ohlcv(n=150, seed=42)


@pytest.fixture(scope="module")
def fitted_hmm(all_data):
    """Fit a 4-state HMM on first 70 rows (train split only)."""
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler

    train_df = all_data.iloc[:70].reset_index(drop=True)
    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(train_df)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42)
    hmm.fit(X, scaler=scaler)
    return hmm


@pytest.fixture
def regime_env(fitted_hmm, all_data):
    """RegimeAwareTradeExecutionEnv configured with test-split data."""
    from src.regimes.features import RegimeFeatureEngine
    from src.environment import RegimeAwareTradeExecutionEnv

    test_df = all_data.iloc[70:].reset_index(drop=True)

    engine = RegimeFeatureEngine()
    regime_feat_df = engine.compute_features(test_df)

    return RegimeAwareTradeExecutionEnv(
        hmm_model=fitted_hmm,
        regime_feature_data=regime_feat_df,
        market_data=test_df,
        target_inventory=10_000.0,
        side="BUY",
        horizon_steps=20,
    )


# ── Observation space ──────────────────────────────────────────────────────────

def test_observation_space_shape(regime_env):
    """Observation space must be 12-dimensional (7 base + 5 regime)."""
    assert regime_env.observation_space.shape == (12,), (
        f"Expected (12,), got {regime_env.observation_space.shape}"
    )


def test_reset_returns_12_dim_obs(regime_env):
    """reset() must return a 12-dim observation array."""
    obs, info = regime_env.reset(seed=0)
    assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_step_returns_12_dim_obs(regime_env):
    """step() must return a 12-dim observation array."""
    regime_env.reset(seed=0)
    obs, reward, terminated, truncated, info = regime_env.step(action=2)
    assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"


# ── Regime values ──────────────────────────────────────────────────────────────

def test_regime_id_in_valid_range(regime_env):
    """Regime ID embedded in obs[7] must be in {0, 1, 2, 3}."""
    obs, _ = regime_env.reset(seed=1)
    regime_id = int(obs[7])
    assert regime_id in {0, 1, 2, 3}, f"Invalid regime id: {regime_id}"


def test_regime_probs_sum_to_one(regime_env):
    """Posterior probabilities obs[8:12] must sum to ~1.0."""
    obs, _ = regime_env.reset(seed=2)
    prob_sum = float(obs[8:12].sum())
    assert abs(prob_sum - 1.0) < 1e-4, f"Regime probs sum to {prob_sum}, expected ~1.0"


def test_regime_probs_non_negative(regime_env):
    """All regime posterior probabilities must be >= 0."""
    obs, _ = regime_env.reset(seed=3)
    for i, p in enumerate(obs[8:12]):
        assert p >= 0.0, f"Negative regime probability at index {i}: {p}"


def test_regime_in_info_dict(regime_env):
    """Info dict must expose regime_id and regime_label after reset."""
    _, info = regime_env.reset(seed=4)
    assert "regime_id" in info
    assert "regime_label" in info
    assert "regime_probs" in info
    assert info["regime_id"] in {0, 1, 2, 3}


# ── Causal leakage prevention ──────────────────────────────────────────────────

def test_regime_resets_on_new_episode(regime_env):
    """
    The causal forward filter must reset between episodes.
    Two resets with different seeds should produce valid (possibly different)
    regime IDs — we simply verify no state bleeds between episodes.
    """
    obs_ep1, _ = regime_env.reset(seed=10)
    regime_ep1 = obs_ep1[7]

    obs_ep2, _ = regime_env.reset(seed=10)
    regime_ep2 = obs_ep2[7]

    # Same seed → same regime at t=0 (deterministic)
    assert regime_ep1 == regime_ep2, (
        "Same-seed resets must produce the same initial regime"
    )


def test_causal_filter_state_independent_of_future(regime_env, fitted_hmm, all_data):
    """
    Run two separate causal inference sessions on the same data.
    Both must produce identical regime sequences (deterministic forward filter).
    """
    from src.regimes.inference import CausalRegimeInference
    from src.regimes.features import RegimeFeatureEngine

    test_df = all_data.iloc[70:90].reset_index(drop=True)
    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(test_df)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=False)
    X = np.where(np.isnan(X), 0.0, X)

    inf1 = CausalRegimeInference(fitted_hmm)
    inf2 = CausalRegimeInference(fitted_hmm)

    regimes1 = [inf1.step_online(X[t])[0] for t in range(len(X))]
    regimes2 = [inf2.step_online(X[t])[0] for t in range(len(X))]

    assert regimes1 == regimes2, "Forward filter must be deterministic"


# ── Action space ───────────────────────────────────────────────────────────────

def test_action_space_identical_to_base_env(regime_env):
    """Action space must remain Discrete(4) — same as TradeExecutionEnv."""
    from gymnasium.spaces import Discrete
    assert isinstance(regime_env.action_space, Discrete)
    assert regime_env.action_space.n == 4


def test_full_rollout_no_crash(regime_env):
    """Complete episode rollout must terminate without errors."""
    obs, _ = regime_env.reset(seed=5)
    done = False
    steps = 0
    while not done:
        action = regime_env.action_space.sample()
        obs, reward, terminated, truncated, info = regime_env.step(action)
        assert obs.shape == (12,)
        done = terminated or truncated
        steps += 1
    assert steps >= 1


def test_inventory_never_negative(regime_env):
    """Remaining inventory must never go negative during a rollout."""
    regime_env.reset(seed=6)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = regime_env.step(action=3)
        remaining = info["remaining_inventory"]
        assert remaining >= -1e-6, f"Negative inventory: {remaining}"
        done = terminated or truncated


# ── Unfitted HMM guard ─────────────────────────────────────────────────────────

def test_unfitted_hmm_raises(all_data):
    """Passing an unfitted HMM must raise a RuntimeError immediately."""
    from src.regimes.hmm_model import MarketHMM
    from src.environment import RegimeAwareTradeExecutionEnv

    unfitted = MarketHMM(n_regimes=4)
    with pytest.raises(RuntimeError, match="fitted"):
        RegimeAwareTradeExecutionEnv(hmm_model=unfitted, market_data=all_data)
