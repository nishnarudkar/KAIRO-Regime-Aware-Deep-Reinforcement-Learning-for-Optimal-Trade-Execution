"""
Tests for the second-round design changes: participation-sized orders, TWAP-relative actions,
the drift-free training reward, new causal regime features and the real-data grid.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import protocol as P
from src.evaluation.scenarios import generate_scenario_data


@pytest.fixture(scope="module")
def split():
    return P.make_split(generate_scenario_data("normal", 3000, seed=5))


class _Const:
    def __init__(self, a):
        self.a = a

    def predict(self, obs, deterministic=True):
        return self.a


# ── Participation-sized orders ────────────────────────────────────────────────

def test_order_size_scales_with_liquidity():
    normal = P.make_split(generate_scenario_data("normal", 3000, seed=5)).full
    thin = P.make_split(generate_scenario_data("low_liquidity", 3000, seed=5)).full
    assert P.window_order_size(thin, 2200) < 0.4 * P.window_order_size(normal, 2200)


def test_order_size_uses_only_history():
    df = generate_scenario_data("normal", 1000, seed=1)
    a = P.window_order_size(df, 600)
    tampered = df.copy()
    tampered.loc[600:, "volume"] *= 50
    assert P.window_order_size(tampered, 600) == pytest.approx(a)


def test_env_and_helper_agree_and_orders_are_fillable(split):
    starts = P.test_window_starts(split)
    env = P.build_env("plain", [split.full], order_participation=P.ORDER_PARTICIPATION)
    env.reset(options={"start_idx": starts[2]})
    assert env.episode_target == pytest.approx(P.window_order_size(split.full, starts[2]))
    for scen in ("low_liquidity", "liquidity_shock"):
        sp = P.make_split(generate_scenario_data(scen, 3000, seed=6))
        res = P.evaluate_baseline_windows(sp.full, P.test_window_starts(sp)[:6],
                                          order_participation=P.ORDER_PARTICIPATION)
        assert (res["fill_rate"] > 0.999).all(), scen        # thin markets no longer make orders unfillable


# ── TWAP-relative actions ─────────────────────────────────────────────────────

def test_default_action_space_is_twap_multiples(split):
    env = P.build_env("plain", [split.full])
    assert env.action_space.n == 5 and env.action_mode == "twap_multiple"
    assert env.action_fractions == P.ACTION_MULTIPLIERS


def test_playing_1x_reproduces_twap_exactly(split):
    starts = P.test_window_starts(split)[:8]
    env = P.build_env("plain", [split.full], order_participation=P.ORDER_PARTICIPATION)
    mine = P.evaluate_agent_windows(_Const(2), env, starts, "1x")           # action 2 == 1.0x TWAP slice
    twap = P.evaluate_baseline_windows(split.full, starts, order_participation=P.ORDER_PARTICIPATION)
    twap = twap[twap.strategy == "TWAP"]
    np.testing.assert_allclose(mine.implementation_shortfall_bps.to_numpy(),
                               twap.implementation_shortfall_bps.to_numpy(), atol=0.02)
    assert (mine.fill_rate > 0.999).all()


def test_pausing_forever_leaves_the_order_unfilled(split):
    env = P.build_env("plain", [split.full], order_participation=P.ORDER_PARTICIPATION)
    out = P.rollout_agent_window(_Const(0), env, P.test_window_starts(split)[0])
    assert out["fill_rate"] == 0.0 and out["terminal_penalty"] > 0


def test_fraction_mode_is_still_available(split):
    env = P.build_env("plain", [split.full], action_mode="fraction")
    assert env.action_space.n == 4 and env.action_mode == "fraction"


def test_last_step_can_finish_the_order(split):
    env = P.build_env("plain", [split.full], order_participation=P.ORDER_PARTICIPATION)
    env.reset(options={"start_idx": P.test_window_starts(split)[1]})
    for _ in range(P.HORIZON_STEPS - 1):
        env.step(0)                                                        # wait until the final step
    assert env.simulator.remaining_inventory > 0
    env.step(2)                                                            # 1x slice at the last step = everything left
    # what remains is only what the 15% participation cap prevented from filling in one bar
    assert env.simulator.executed_inventory > 0


# ── Drift-free reward ─────────────────────────────────────────────────────────

def test_drift_free_reward_removes_price_noise_but_keeps_costs():
    sp = P.make_split(generate_scenario_data("stress", 3000, seed=7))
    starts = P.test_window_starts(sp)[:20]
    stats = {}
    for flag in (False, True):
        env = P.build_env("plain", [sp.full], order_participation=P.ORDER_PARTICIPATION, drift_free_reward=flag)
        rets = []
        for i, s in enumerate(starts):
            env.reset(seed=i, options={"start_idx": s})
            done, tot = False, 0.0
            while not done:
                _, r, te, tr, _ = env.step(2)
                tot += r
                done = te or tr
            rets.append(tot)
        stats[flag] = (np.mean(rets), np.std(rets))
    assert stats[True][1] < 0.2 * stats[False][1]          # far less noisy
    assert stats[True][0] < 0                              # still a cost


def test_drift_free_flag_does_not_change_reported_shortfall(split):
    starts = P.test_window_starts(split)[:4]
    outs = []
    for flag in (False, True):
        env = P.build_env("plain", [split.full], order_participation=P.ORDER_PARTICIPATION, drift_free_reward=flag)
        outs.append(P.evaluate_agent_windows(_Const(2), env, starts, "x").implementation_shortfall_bps.to_numpy())
    np.testing.assert_allclose(outs[0], outs[1])           # evaluation is always the true arrival-price IS


# ── Regime features ───────────────────────────────────────────────────────────

def test_new_regime_features_are_causal(split):
    df = split.full.iloc[:600].copy()
    base = P.compute_regime_features(df)
    fut = df.copy()
    fut.loc[400:, ["close", "high", "low", "volume", "bid", "ask", "spread"]] *= 3.0
    changed = P.compute_regime_features(fut)
    for col in P.HMM_FEATURES:
        np.testing.assert_allclose(base[col].iloc[:400].to_numpy(), changed[col].iloc[:400].to_numpy(),
                                   equal_nan=True, err_msg=col)


def test_hmm_features_used_by_protocol(split):
    hmm = P.fit_hmm([split.train])
    assert list(hmm.feature_names) == P.HMM_FEATURES


# ── Real-data grid ────────────────────────────────────────────────────────────

def test_real_data_grid_is_regular_and_safe():
    from scripts.evaluate_real_data import regular_grid

    rng = np.random.default_rng(0)
    ts = pd.to_datetime(["2026-09-01 14:00", "2026-09-01 14:05", "2026-09-01 15:30", "2026-09-01 18:00",
                         "2026-09-02 14:10", "2026-09-02 16:00", "2026-09-02 17:45"], utc=True)
    n = 60
    ts = pd.DatetimeIndex(sorted(pd.to_datetime(
        [pd.Timestamp("2026-09-01 13:30", tz="UTC") + pd.Timedelta(minutes=int(m)) for m in rng.choice(380, n, replace=False)]
        + [pd.Timestamp("2026-09-02 13:30", tz="UTC") + pd.Timedelta(minutes=int(m)) for m in rng.choice(380, n, replace=False)])))
    raw = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0 + rng.normal(0, .1, len(ts)),
                        "volume": rng.integers(10, 500, len(ts)).astype(float)})
    g = regular_grid(raw)
    assert len(g) % 390 == 0 and len(g) == 780
    assert g["price"].notna().all() and (g["volume"] >= 0).all()
    assert (g["ask"] > g["bid"]).all()
