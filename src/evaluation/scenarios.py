"""
src/evaluation/scenarios.py — Synthetic Market Scenario Generators

Every scenario is a *regime-switching* one-minute market. A latent Markov chain
over four regimes (low-vol, normal, high-vol, stress) drives, per bar:

  - the log-return volatility,
  - the quoted bid-ask spread,
  - traded volume.

Scales are calibrated to a liquid US large-cap at one-minute resolution
(per-bar return volatility of roughly 0.02%-0.3%, spreads of ~1-10 bps,
~60k shares per minute in a normal regime) instead of the previous
5%-per-minute random walk. Price drift is zero in every scenario, so
implementation shortfall measures execution quality rather than luck with
the price path.

The regime path is returned in the ``true_regime`` column. Agents and the
HMM never see it; it exists so regime detection can be validated.

Scenarios
---------
  normal            calm-to-normal market with occasional high-vol spells
  high_volatility   mostly high-volatility conditions
  low_liquidity     normal regimes with ~1/4 of the volume and 3x the spread
  stress            stress-dominated market
  regime_transition calm for the first 60% of the series, turbulent afterwards
  liquidity_shock   normal for the first 60%, then volume collapses to 10%
                    and spreads triple

All generated data is synthetic and deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── Regime definitions ─────────────────────────────────────────────────────────

REGIME_NAMES = ["Low Volatility", "Normal", "High Volatility", "Stress"]

# Per-bar (one minute) parameters for each latent regime.
REGIME_SIGMA = np.array([0.00025, 0.00050, 0.00120, 0.00300])   # log-return std
REGIME_SPREAD = np.array([0.010, 0.020, 0.050, 0.150])          # mean quoted spread ($)
REGIME_VOLUME = np.array([0.80, 1.00, 1.40, 1.80])              # volume multiplier

SWITCH_FRACTION = 0.60   # regime_transition / liquidity_shock switch point


def _persistent(stay: np.ndarray, move: np.ndarray) -> np.ndarray:
    """Build a transition matrix with per-state stay probability and off-diagonal weights."""
    P = np.zeros((4, 4))
    for i in range(4):
        w = move[i].astype(float).copy()
        w[i] = 0.0
        w = w / w.sum() * (1.0 - stay[i])
        P[i] = w
        P[i, i] = stay[i]
    return P


# Transition matrices (one row per current regime, rows sum to 1). Stay
# probabilities give expected regime durations of 100-200 bars (calm) and
# 30-70 bars (extreme), so a 30-bar execution window is usually inside one regime.
_CALM_P = _persistent(
    stay=np.array([0.990, 0.992, 0.975, 0.950]),
    move=np.array([[0, 6, 1, 0.2], [3, 0, 2, 0.2], [1, 4, 0, 2], [0.5, 2, 5, 0]]),
)
_MIXED_P = _persistent(
    stay=np.array([0.985, 0.988, 0.985, 0.975]),
    move=np.array([[0, 3, 2, 1], [2, 0, 3, 1], [1, 3, 0, 3], [1, 2, 4, 0]]),
)
_HIGHVOL_P = _persistent(
    stay=np.array([0.970, 0.975, 0.992, 0.985]),
    move=np.array([[0, 3, 4, 1], [1, 0, 4, 1], [0.5, 2, 0, 3], [0.5, 1, 6, 0]]),
)
_STRESS_P = _persistent(
    stay=np.array([0.960, 0.970, 0.985, 0.992]),
    move=np.array([[0, 2, 4, 2], [0.5, 0, 4, 2], [0.5, 1, 0, 6], [0.2, 0.5, 5, 0]]),
)


@dataclass
class ScenarioConfig:
    """Parameters defining a synthetic market scenario."""
    name: str
    description: str
    transition: np.ndarray
    initial: np.ndarray
    volume_mean: float = 60_000.0     # mean bar volume in the normal regime (shares)
    volume_scale: float = 1.0
    spread_scale: float = 1.0
    # Optional second phase entered at SWITCH_FRACTION of the series.
    switch_transition: Optional[np.ndarray] = None
    switch_initial: Optional[np.ndarray] = None
    switch_volume_scale: Optional[float] = None
    switch_spread_scale: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


_UNIFORM_CALM = np.array([0.45, 0.45, 0.10, 0.0])

SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        name="normal",
        description="Calm-to-normal conditions with occasional high-volatility spells.",
        transition=_CALM_P,
        initial=_UNIFORM_CALM,
    ),
    "high_volatility": ScenarioConfig(
        name="high_volatility",
        description="Mostly high-volatility conditions with wider spreads.",
        transition=_HIGHVOL_P,
        initial=np.array([0.0, 0.2, 0.7, 0.1]),
    ),
    "low_liquidity": ScenarioConfig(
        name="low_liquidity",
        description="Thin market: about a quarter of normal volume and triple the spread.",
        transition=_CALM_P,
        initial=_UNIFORM_CALM,
        volume_scale=0.25,
        spread_scale=3.0,
    ),
    "stress": ScenarioConfig(
        name="stress",
        description="Stress-dominated market: extreme volatility, very wide spreads.",
        transition=_STRESS_P,
        initial=np.array([0.0, 0.0, 0.3, 0.7]),
    ),
    "regime_transition": ScenarioConfig(
        name="regime_transition",
        description="Calm for the first 60% of the series, turbulent (high-vol / stress) afterwards.",
        transition=_CALM_P,
        initial=_UNIFORM_CALM,
        switch_transition=_STRESS_P,
        switch_initial=np.array([0.0, 0.0, 1.0, 0.0]),
    ),
    "liquidity_shock": ScenarioConfig(
        name="liquidity_shock",
        description="Normal conditions, then volume collapses to 10% and spreads triple.",
        transition=_CALM_P,
        initial=_UNIFORM_CALM,
        switch_transition=_CALM_P,
        switch_initial=np.array([0.0, 1.0, 0.0, 0.0]),
        switch_volume_scale=0.10,
        switch_spread_scale=3.0,
    ),
}


# ── Generation ─────────────────────────────────────────────────────────────────

def _sample_regimes(
    rng: np.random.Generator,
    n: int,
    cfg: ScenarioConfig,
) -> np.ndarray:
    """Sample the latent regime path (Markov chain, optional switch of dynamics)."""
    regimes = np.empty(n, dtype=np.int64)
    switch_at = int(n * SWITCH_FRACTION) if cfg.switch_transition is not None else n

    state = int(rng.choice(4, p=cfg.initial))
    P = cfg.transition
    for t in range(n):
        if t == switch_at:
            P = cfg.switch_transition
            state = int(rng.choice(4, p=cfg.switch_initial))
        regimes[t] = state
        state = int(rng.choice(4, p=P[state]))
    return regimes


BARS_PER_DAY = 390   # 09:30-16:00 one-minute bars


def _trading_timestamps(n: int) -> pd.DatetimeIndex:
    """One-minute timestamps over consecutive business days, 390 bars per day."""
    days = pd.bdate_range("2025-06-02", periods=int(np.ceil(n / BARS_PER_DAY)) + 1)
    idx = np.arange(n)
    return pd.DatetimeIndex(
        days[idx // BARS_PER_DAY] + pd.Timedelta(hours=9, minutes=30) + pd.to_timedelta(idx % BARS_PER_DAY, unit="min")
    )


def _causal_rolling_vol(log_ret: np.ndarray, fallback: float, window: int = 20) -> np.ndarray:
    """Backward-looking rolling std of returns; only past/current returns are used."""
    s = pd.Series(log_ret)
    vol = s.rolling(window=window, min_periods=5).std()
    return vol.fillna(fallback).clip(lower=1e-5).to_numpy()


def generate_scenario_data(
    scenario_name: str,
    n_steps: int = 200,
    seed: int = 42,
    base_price: float = 150.0,
) -> pd.DataFrame:
    """
    Generate synthetic one-minute market data for a named scenario.

    Args:
        scenario_name: Key from SCENARIOS.
        n_steps: Number of one-minute bars.
        seed: Random seed (fully deterministic).
        base_price: Starting price.

    Returns:
        DataFrame with columns
        [timestamp, open, high, low, close, price, bid, ask, volume, spread,
         volatility, true_regime].
        ``volatility`` is a causal 20-bar realized volatility of log returns.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Valid options: {list(SCENARIOS.keys())}"
        )

    cfg = SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)
    n = int(n_steps)

    regimes = _sample_regimes(rng, n, cfg)
    sigma = REGIME_SIGMA[regimes]

    # Price path: zero drift, regime-dependent volatility.
    log_ret = rng.normal(0.0, sigma)
    log_ret[0] = 0.0
    close = base_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[base_price], close[:-1]])

    # Intrabar range scales with the regime volatility.
    up = np.abs(rng.normal(0.0, sigma * 0.5))
    dn = np.abs(rng.normal(0.0, sigma * 0.5))
    high = np.maximum(open_, close) * np.exp(up)
    low = np.minimum(open_, close) * np.exp(-dn)

    # Volume: lognormal around the regime mean with a mild intraday U-shape.
    idx = np.arange(n)
    phase = (idx % 390) / 389.0
    u_shape = 1.0 + 0.5 * ((2.0 * phase - 1.0) ** 2 - 1.0 / 3.0)
    vol_scale = np.full(n, cfg.volume_scale)
    spread_scale = np.full(n, cfg.spread_scale)
    if cfg.switch_transition is not None:
        s_at = int(n * SWITCH_FRACTION)
        if cfg.switch_volume_scale is not None:
            vol_scale[s_at:] = cfg.switch_volume_scale
        if cfg.switch_spread_scale is not None:
            spread_scale[s_at:] = cfg.switch_spread_scale
    mean_volume = cfg.volume_mean * REGIME_VOLUME[regimes] * vol_scale * u_shape
    volume = np.maximum(500.0, mean_volume * rng.lognormal(mean=-0.03, sigma=0.25, size=n))

    # Quoted spread: regime-dependent, never below one cent.
    spread = np.maximum(
        0.01,
        REGIME_SPREAD[regimes] * spread_scale * rng.lognormal(mean=-0.02, sigma=0.20, size=n),
    )
    bid = close - spread / 2.0
    ask = close + spread / 2.0

    volatility = _causal_rolling_vol(log_ret, fallback=float(REGIME_SIGMA[1]))

    return pd.DataFrame({
        "timestamp":   _trading_timestamps(n),
        "open":        open_,
        "high":        high,
        "low":         low,
        "close":       close,
        "price":       close,
        "bid":         bid,
        "ask":         ask,
        "volume":      volume,
        "spread":      spread,
        "volatility":  volatility,
        "true_regime": regimes,
    })
