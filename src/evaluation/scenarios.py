"""
src/evaluation/scenarios.py — Market Scenario Generators

Generates synthetic market data for different market condition scenarios.
All data is deterministic (fixed seeds) for reproducibility.

Scenarios:
  1. normal          — typical intraday conditions
  2. high_volatility — elevated price movement and wide spreads
  3. low_liquidity   — low volume, large spread, high impact
  4. stress          — extreme volatility, very wide spreads
  5. regime_transition — starts calm, transitions to high volatility mid-episode
  6. liquidity_shock — sudden volume collapse at mid-horizon

IMPORTANT: All generated data is clearly labeled as synthetic.
           No real market data is fabricated or presented as real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    """Parameters defining a synthetic market scenario."""
    name: str
    description: str
    price_drift: float        # per-step drift in price
    price_vol: float          # per-step price std dev
    volume_mean: float        # mean bar volume (shares)
    volume_std: float         # std dev of bar volume
    spread_mean: float        # mean bid-ask spread ($)
    spread_std: float         # std dev of spread
    vol_mean: float           # mean realized volatility
    vol_std: float            # std dev of realized volatility


SCENARIOS: Dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        name="normal",
        description="Typical intraday conditions. Moderate volatility, normal liquidity.",
        price_drift=0.0,
        price_vol=0.05,
        volume_mean=60_000,
        volume_std=10_000,
        spread_mean=0.04,
        spread_std=0.01,
        vol_mean=0.0015,
        vol_std=0.0003,
    ),
    "high_volatility": ScenarioConfig(
        name="high_volatility",
        description="Elevated price swings. Wider spreads. Higher market impact.",
        price_drift=0.0,
        price_vol=0.20,
        volume_mean=80_000,
        volume_std=20_000,
        spread_mean=0.10,
        spread_std=0.03,
        vol_mean=0.005,
        vol_std=0.001,
    ),
    "low_liquidity": ScenarioConfig(
        name="low_liquidity",
        description="Thin market. Low volume, large spread, high impact per share.",
        price_drift=0.0,
        price_vol=0.06,
        volume_mean=15_000,
        volume_std=3_000,
        spread_mean=0.15,
        spread_std=0.05,
        vol_mean=0.002,
        vol_std=0.0005,
    ),
    "stress": ScenarioConfig(
        name="stress",
        description="Extreme conditions. Very high volatility, very wide spreads.",
        price_drift=-0.02,      # mild downward drift (selling pressure)
        price_vol=0.50,
        volume_mean=100_000,
        volume_std=40_000,
        spread_mean=0.30,
        spread_std=0.10,
        vol_mean=0.012,
        vol_std=0.003,
    ),
    "regime_transition": ScenarioConfig(
        name="regime_transition",
        description="First half: calm (low vol). Second half: high volatility. Tests adaptation.",
        price_drift=0.0,
        price_vol=0.05,         # overridden per-segment in generator
        volume_mean=60_000,
        volume_std=10_000,
        spread_mean=0.04,
        spread_std=0.01,
        vol_mean=0.0015,
        vol_std=0.0003,
    ),
    "liquidity_shock": ScenarioConfig(
        name="liquidity_shock",
        description="Normal conditions then sudden volume collapse at midpoint.",
        price_drift=0.0,
        price_vol=0.05,
        volume_mean=60_000,
        volume_std=10_000,
        spread_mean=0.04,
        spread_std=0.01,
        vol_mean=0.0015,
        vol_std=0.0003,
    ),
}


def generate_scenario_data(
    scenario_name: str,
    n_steps: int = 200,
    seed: int = 42,
    base_price: float = 150.0,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV market data for a named scenario.

    Args:
        scenario_name: Key from SCENARIOS dict.
        n_steps: Number of 1-minute bars.
        seed: Random seed for full reproducibility.
        base_price: Starting price.

    Returns:
        DataFrame with columns:
          [timestamp, open, high, low, close, price, volume, spread, volatility]
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Valid options: {list(SCENARIOS.keys())}"
        )

    cfg = SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-06-01 09:30", periods=n_steps, freq="1min")

    if scenario_name == "regime_transition":
        return _make_regime_transition(rng, dates, base_price, n_steps)

    if scenario_name == "liquidity_shock":
        return _make_liquidity_shock(rng, dates, base_price, n_steps, cfg)

    # Standard scenario
    returns = rng.normal(cfg.price_drift, cfg.price_vol, n_steps)
    close = base_price * np.exp(np.cumsum(returns))
    high = close + rng.uniform(0.01, cfg.spread_mean * 2, n_steps)
    low = close - rng.uniform(0.01, cfg.spread_mean * 2, n_steps)
    low = np.minimum(low, close)   # enforce low <= close
    high = np.maximum(high, close) # enforce high >= close

    volume = np.maximum(
        1_000,
        rng.normal(cfg.volume_mean, cfg.volume_std, n_steps)
    ).astype(float)

    spread = np.maximum(
        0.01,
        rng.normal(cfg.spread_mean, cfg.spread_std, n_steps)
    )

    volatility = np.maximum(
        0.0001,
        rng.normal(cfg.vol_mean, cfg.vol_std, n_steps)
    )

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


def _make_regime_transition(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    base_price: float,
    n: int,
) -> pd.DataFrame:
    """Split: first half calm, second half high-vol."""
    half = n // 2
    cfg_calm = SCENARIOS["normal"]
    cfg_hv   = SCENARIOS["high_volatility"]

    ret_calm = rng.normal(0.0, cfg_calm.price_vol, half)
    ret_hv   = rng.normal(0.0, cfg_hv.price_vol, n - half)
    returns  = np.concatenate([ret_calm, ret_hv])
    close    = base_price * np.exp(np.cumsum(returns))

    vol_calm = rng.normal(cfg_calm.vol_mean, cfg_calm.vol_std, half).clip(0.0001)
    vol_hv   = rng.normal(cfg_hv.vol_mean, cfg_hv.vol_std, n - half).clip(0.0001)
    volatility = np.concatenate([vol_calm, vol_hv])

    sp_calm = rng.normal(cfg_calm.spread_mean, cfg_calm.spread_std, half).clip(0.01)
    sp_hv   = rng.normal(cfg_hv.spread_mean, cfg_hv.spread_std, n - half).clip(0.01)
    spread  = np.concatenate([sp_calm, sp_hv])

    vol_calm_v = rng.normal(cfg_calm.volume_mean, cfg_calm.volume_std, half).clip(1_000)
    vol_hv_v   = rng.normal(cfg_hv.volume_mean, cfg_hv.volume_std, n - half).clip(1_000)
    volume = np.concatenate([vol_calm_v, vol_hv_v])

    high = close + spread / 2
    low  = (close - spread / 2).clip(0.01)

    return pd.DataFrame({
        "timestamp":  dates,
        "open":       close, "high": high, "low": low,
        "close":      close, "price": close,
        "volume":     volume, "spread": spread, "volatility": volatility,
    })


def _make_liquidity_shock(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    base_price: float,
    n: int,
    cfg: ScenarioConfig,
) -> pd.DataFrame:
    """Normal conditions, then volume collapses to 10% at midpoint."""
    returns = rng.normal(cfg.price_drift, cfg.price_vol, n)
    close = base_price * np.exp(np.cumsum(returns))
    high = close + rng.uniform(0.01, cfg.spread_mean * 2, n)
    low  = np.minimum(close - rng.uniform(0.01, cfg.spread_mean * 2, n), close)
    high = np.maximum(high, close)

    volume = rng.normal(cfg.volume_mean, cfg.volume_std, n).clip(1_000)
    # Shock: cut volume to 10% from midpoint onwards
    volume[n // 2:] *= 0.10

    spread = rng.normal(cfg.spread_mean, cfg.spread_std, n).clip(0.01)
    spread[n // 2:] *= 3.0   # spreads widen during liquidity shock

    volatility = rng.normal(cfg.vol_mean, cfg.vol_std, n).clip(0.0001)

    return pd.DataFrame({
        "timestamp":  dates,
        "open":       close, "high": high, "low": low,
        "close":      close, "price": close,
        "volume":     volume, "spread": spread, "volatility": volatility,
    })
