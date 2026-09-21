"""
src/api/engine.py — Execution engine used by the API routers

Everything here goes through ``src.evaluation.protocol`` so the API, the
research experiments and the training script share one definition of a market,
a window and a rollout. Learned policies are loaded from disk by the registry;
nothing is trained inside a request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.agents import registry
from src.evaluation import protocol as P
from src.evaluation.scenarios import generate_scenario_data

BASELINES = registry.BASELINE_POLICIES
RL_POLICIES = registry.RL_POLICIES


@lru_cache(maxsize=16)
def _market(scenario: str, seed: int) -> Tuple[pd.DataFrame, P.SeriesSplit]:
    df = generate_scenario_data(scenario, n_steps=P.N_BARS, seed=seed)
    return df, P.make_split(df)


def get_market(scenario: str, seed: int) -> Tuple[pd.DataFrame, P.SeriesSplit]:
    """Deterministic synthetic market for (scenario, seed) plus its train/test split."""
    return _market(scenario, seed)


def test_starts(split: P.SeriesSplit, horizon: int) -> List[int]:
    """Non-overlapping out-of-sample window starts for the requested horizon."""
    return P.test_window_starts(split, horizon=horizon)


def pick_window(starts: List[int], seed: int) -> int:
    """Deterministically choose one out-of-sample window from the seed."""
    return starts[seed % len(starts)]


def run_policy(
    policy: str,
    df: pd.DataFrame,
    start: int,
    horizon: int,
    side: str,
    quantity: float,
    record_obs: bool = False,
) -> Dict[str, Any]:
    """
    Execute one policy on one out-of-sample window.

    Returns the rollout dict from the protocol module plus ``price_trajectory``.
    Raises registry.ModelUnavailable for a learned policy without a checkpoint.
    """
    if policy in BASELINES:
        out = P.rollout_baseline_window(policy, df, start, horizon=horizon,
                                        target_inventory=quantity, side=side)
    elif policy in RL_POLICIES:
        lp = registry.load_policy(policy)
        env = P.build_env(lp.spec["kind"], [df], lp.hmm, horizon=horizon,
                          target_inventory=quantity, side=side)
        out = P.rollout_agent_window(lp.agent, env, start, seed=0, record_obs=record_obs)
    else:
        raise ValueError(f"Unsupported policy '{policy}'.")
    out["price_trajectory"] = [float(x) for x in df["price"].iloc[start:start + horizon]]
    return out


def action_counts(actions: List[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in actions:
        counts[str(int(a))] = counts.get(str(int(a)), 0) + 1
    return counts


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)
