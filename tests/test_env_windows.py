"""
Tests for episode windows, causality of the observation and reward scaling.
"""

import numpy as np
import pytest

from src.environment import TradeExecutionEnv, RegimeAwareTradeExecutionEnv
from src.environment.rewards import DEFAULT_REWARD_SCALE
from src.evaluation import protocol as P
from src.evaluation.scenarios import generate_scenario_data


@pytest.fixture(scope="module")
def df():
    return generate_scenario_data("normal", 1500, seed=21)


def test_default_env_still_replays_the_first_window(df):
    env = TradeExecutionEnv(market_data=df, horizon_steps=30)
    env.reset(seed=0)
    assert env.window_start == 0
    env.reset(seed=1)
    assert env.window_start == 0


def test_random_start_varies_and_respects_range(df):
    env = TradeExecutionEnv(market_data=df, horizon_steps=30, random_start=True, start_range=(100, 900))
    starts = []
    for s in range(40):
        env.reset(seed=s)
        starts.append(env.window_start)
    assert len(set(starts)) > 20
    assert min(starts) >= 100 and max(starts) <= 900


def test_start_idx_option_and_arrival_price(df):
    env = TradeExecutionEnv(market_data=df, horizon_steps=30)
    env.reset(seed=0, options={"start_idx": 500})
    assert env.window_start == 500
    assert env.simulator.arrival_price == pytest.approx(df["price"].iloc[500])


def test_episodes_are_truly_different_paths(df):
    """Training on a single replayed path was the original flaw."""
    env = TradeExecutionEnv(market_data=df, horizon_steps=30, random_start=True, start_range=(60, 1400))
    prices = []
    for s in range(5):
        env.reset(seed=s)
        prices.append(env.simulator.arrival_price)
    assert len(set(np.round(prices, 6))) == 5


def test_multiple_datasets_are_sampled(df):
    other = generate_scenario_data("stress", 1500, seed=22)
    env = TradeExecutionEnv(market_data=[df, other], horizon_steps=30, random_start=True)
    seen = set()
    for s in range(30):
        env.reset(seed=s)
        seen.add(env.dataset_idx)
    assert seen == {0, 1}


def test_observation_does_not_depend_on_the_future(df):
    """Volume reference and previous price use only bars before the window."""
    env = TradeExecutionEnv(market_data=df, horizon_steps=30)
    obs_a, _ = env.reset(seed=0, options={"start_idx": 400})
    future = df.copy()
    future.loc[430:, ["volume", "price", "bid", "ask"]] *= 3.0      # beyond the 30-bar window
    env2 = TradeExecutionEnv(market_data=future, horizon_steps=30)
    obs_b, _ = env2.reset(seed=0, options={"start_idx": 400})
    np.testing.assert_allclose(obs_a, obs_b)


def test_observation_scale_is_well_conditioned(df):
    env = TradeExecutionEnv(market_data=df, horizon_steps=30, random_start=True, start_range=(60, 1400))
    seen = []
    for s in range(20):
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            seen.append(obs)
            obs, _, term, trunc, _ = env.step(1)
            done = term or trunc
    arr = np.array(seen)
    # The old observation mixed values of ~1e-3 and ~1e6; now everything is O(1)-O(10).
    assert np.abs(arr).max() < 50.0
    assert np.isfinite(arr).all()


def test_reward_is_expressed_in_bps_scale(df):
    env = TradeExecutionEnv(market_data=df, horizon_steps=30)
    assert env.reward_calculator.reward_scale == DEFAULT_REWARD_SCALE
    env.reset(seed=0)
    _, reward, *_ = env.step(3)
    assert abs(reward) > 1e-3       # previously ~1e-4, starving gradient-based learners


def _regime_env(df):
    hmm = P.fit_hmm([df.iloc[:1000]])
    feats = P.compute_regime_features(df)
    return RegimeAwareTradeExecutionEnv(hmm_model=hmm, regime_feature_data=feats, market_data=df,
                                        horizon_steps=30), feats


def test_regime_filter_is_warm_started(df):
    env, _ = _regime_env(df)
    obs, _ = env.reset(seed=0, options={"start_idx": 600})
    probs = obs[8:12]
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert probs.max() > 0.3                       # informative, not the uniform prior 0.25 each
    obs0, _ = env.reset(seed=0, options={"start_idx": 0})
    assert obs0.shape == (12,)


def test_regime_belief_is_causal(df):
    """Altering bars after the current step must not change regime probabilities up to it."""
    env, feats = _regime_env(df)
    env.reset(seed=0, options={"start_idx": 600})
    a = []
    for _ in range(10):
        o, *_ = env.step(1)
        a.append(o[8:12].copy())

    df2 = df.copy()
    df2.loc[612:, ["high", "low", "close", "price", "volume"]] *= 1.5
    env2 = RegimeAwareTradeExecutionEnv(hmm_model=env.hmm_model, regime_feature_data=P.compute_regime_features(df2),
                                        market_data=df2, horizon_steps=30)
    env2.reset(seed=0, options={"start_idx": 600})
    b = []
    for _ in range(10):
        o, *_ = env2.step(1)
        b.append(o[8:12].copy())
    np.testing.assert_allclose(np.array(a)[:9], np.array(b)[:9], atol=1e-6)
