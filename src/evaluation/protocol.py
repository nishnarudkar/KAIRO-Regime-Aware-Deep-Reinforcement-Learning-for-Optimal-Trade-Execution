"""
src/evaluation/protocol.py — Shared training / evaluation protocol

One place defines how KAIRO agents are trained and evaluated so that the
research experiments, the model-training script and the API all agree.

Protocol
--------
  * One long synthetic series per (scenario, seed).
  * Chronological split: the first 70% is *train*, the last 30% is *test*.
  * The HMM is fitted on train features only.
  * Agents train on **random 30-bar windows** of the train region (many
    different market paths, not one replayed path).
  * Evaluation uses **non-overlapping windows** covering the test region.
    Every strategy is run on exactly the same windows, so comparisons are
    paired. The history preceding a window is available to strategies as past
    information only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_STEPS = 30
TARGET_INVENTORY = 100_000.0
SIDE = "BUY"
TRAIN_RATIO = 0.70
WARMUP_BARS = 60          # bars of history required before any episode start
N_BARS = 3000             # length of each synthetic series (~7.7 trading days)
N_REGIMES = 4

# Strategy names used in result tables (kept stable across the code base).
NAME_TWAP, NAME_VWAP, NAME_POV = "TWAP", "VWAP", "POV"
NAME_DQN, NAME_RA_DQN, NAME_SH_DQN = "DQN", "Regime-Aware DQN", "DQN (shuffled regime)"
NAME_PPO, NAME_RA_PPO, NAME_SH_PPO = "PPO", "Regime-Aware PPO", "PPO (shuffled regime)"
BASELINE_NAMES = [NAME_TWAP, NAME_VWAP, NAME_POV]

# (algorithm, env kind) for each learned strategy.
LEARNED_SPECS: Dict[str, Tuple[str, str]] = {
    NAME_DQN: ("dqn", "plain"),
    NAME_RA_DQN: ("dqn", "regime"),
    NAME_SH_DQN: ("dqn", "shuffled"),
    NAME_PPO: ("ppo", "plain"),
    NAME_RA_PPO: ("ppo", "regime"),
    NAME_SH_PPO: ("ppo", "shuffled"),
}


# ── Data splitting ────────────────────────────────────────────────────────────

@dataclass
class SeriesSplit:
    """A full series with its chronological train / test boundary."""
    full: pd.DataFrame
    cut: int

    @property
    def train(self) -> pd.DataFrame:
        return self.full.iloc[: self.cut].reset_index(drop=True)

    @property
    def n(self) -> int:
        return len(self.full)


def make_split(df: pd.DataFrame, ratio: float = TRAIN_RATIO) -> SeriesSplit:
    """Chronological split. Never shuffled."""
    return SeriesSplit(full=df.reset_index(drop=True), cut=int(len(df) * ratio))


def train_start_range(cut: int, horizon: int = HORIZON_STEPS) -> Tuple[int, int]:
    """Inclusive range of admissible episode starts inside the train region."""
    return (min(WARMUP_BARS, max(0, cut - horizon)), max(0, cut - horizon))


def test_window_starts(split: SeriesSplit, horizon: int = HORIZON_STEPS,
                       max_windows: Optional[int] = None) -> List[int]:
    """Non-overlapping test windows; each window lies entirely in the test region."""
    starts = list(range(split.cut, split.n - horizon + 1, horizon))
    return starts[:max_windows] if max_windows else starts


# ── HMM ───────────────────────────────────────────────────────────────────────

def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.regimes.features import RegimeFeatureEngine
    return RegimeFeatureEngine().compute_features(df)


def fit_hmm(train_dfs: Sequence[pd.DataFrame], random_state: int = 42):
    """
    Fit a 4-state Gaussian HMM on the pooled *training* features.

    Returns None when there is too little data to fit.
    """
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler

    engine = RegimeFeatureEngine()
    blocks = []
    for d in train_dfs:
        _, X = engine.extract_feature_matrix(engine.compute_features(d), drop_na=True)
        blocks.append(X)
    X = np.vstack(blocks)
    if len(X) < 50:
        return None
    hmm = MarketHMM(n_regimes=N_REGIMES, random_state=random_state)
    try:
        hmm.fit(X, scaler=RegimeFeatureScaler(method="robust"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("HMM fit failed: %s", exc)
        return None
    return hmm


# ── Environments ──────────────────────────────────────────────────────────────

def build_env(
    kind: str,
    datasets: Sequence[pd.DataFrame],
    hmm=None,
    *,
    random_start: bool = False,
    start_range: Optional[Tuple[int, int]] = None,
    horizon: int = HORIZON_STEPS,
    target_inventory: float = TARGET_INVENTORY,
    side: str = SIDE,
    shuffle_seed: int = 1000,
):
    """
    Build an environment of the requested kind over one or more datasets.

    kind: "plain" (7-dim), "regime" (12-dim, causal HMM regime) or
          "shuffled" (12-dim, uninformative regime noise; ablation control).
    """
    from src.environment import TradeExecutionEnv, RegimeAwareTradeExecutionEnv
    from src.evaluation.ablation import ShuffledRegimeEnv

    data = list(datasets) if len(datasets) > 1 else datasets[0]
    common = dict(
        target_inventory=target_inventory, side=side, horizon_steps=horizon,
        random_start=random_start, start_range=start_range,
    )
    if kind == "plain":
        return TradeExecutionEnv(market_data=data, **common)
    if kind == "shuffled":
        return ShuffledRegimeEnv(market_data=data, shuffle_seed=shuffle_seed, **common)
    if kind == "regime":
        if hmm is None:
            raise ValueError("A fitted HMM is required for regime-aware environments.")
        feats = [compute_regime_features(d) for d in datasets]
        return RegimeAwareTradeExecutionEnv(
            hmm_model=hmm,
            regime_feature_data=feats if len(feats) > 1 else feats[0],
            market_data=data, **common,
        )
    raise ValueError(f"Unknown env kind '{kind}'")


# ── Agents ────────────────────────────────────────────────────────────────────

# Hyperparameters used by every experiment and served model. config/dqn.yaml and
# config/ppo.yaml document the same values; tests/test_config_consistency.py fails if they drift.
DQN_HPARAMS: Dict[str, Any] = dict(
    learning_rate=3e-4, buffer_size=100_000, batch_size=64, gamma=0.99, train_freq=4,
    target_update_interval=1_000, exploration_fraction=0.3,
    exploration_initial_eps=1.0, exploration_final_eps=0.05,
)
PPO_HPARAMS: Dict[str, Any] = dict(
    learning_rate=3e-4, n_steps=1_024, batch_size=128, n_epochs=10, gamma=0.99,
    gae_lambda=0.95, clip_range=0.2,
)


def make_agent(algo: str, env, seed: int, timesteps: int):
    """Construct an untrained DQN / PPO agent with the documented hyperparameters."""
    if algo == "dqn":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent(env=env, seed=seed, verbose=0,
                        learning_starts=min(1_000, max(200, timesteps // 20)), **DQN_HPARAMS)
    if algo == "ppo":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent(env=env, seed=seed, verbose=0, **PPO_HPARAMS)
    raise ValueError(f"Unknown algorithm '{algo}'")


def train_agent(algo: str, kind: str, train_dfs: Sequence[pd.DataFrame], hmm, seed: int,
                timesteps: int, cut: Optional[int] = None):
    """Train an agent on random windows of the train region(s). Returns the agent."""
    n = min(len(d) for d in train_dfs)
    rng_range = train_start_range(cut if cut is not None else n)
    env = build_env(kind, train_dfs, hmm, random_start=True, start_range=rng_range,
                    shuffle_seed=seed + 1000)
    agent = make_agent(algo, env, seed, timesteps)
    agent.train(total_timesteps=timesteps)
    return agent


# ── Window evaluation ─────────────────────────────────────────────────────────

_METRIC_FIELDS = (
    "implementation_shortfall_bps", "implementation_shortfall", "total_execution_cost",
    "total_impact_cost", "total_transaction_fees", "terminal_penalty", "fill_rate",
    "vwap_slippage_bps", "average_execution_price", "arrival_price", "vwap_market_price",
    "executed_inventory", "remaining_inventory",
)


def rollout_agent_window(agent, env, start: int, seed: int = 0, dataset_idx: int = 0,
                         record_obs: bool = False) -> Dict[str, Any]:
    """Run a trained agent greedily on one window and return metrics + trajectories."""
    obs, info = env.reset(seed=seed, options={"start_idx": start, "dataset_idx": dataset_idx})
    observations: List[List[float]] = []
    inv = [float(env.simulator.remaining_inventory)]
    actions: List[int] = []
    regimes: List[int] = []
    if "regime_id" in info:
        regimes.append(int(info["regime_id"]))
    done = False
    while not done:
        if record_obs:
            observations.append([float(v) for v in obs])
        action = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        inv.append(float(info["remaining_inventory"]))
        actions.append(int(action))
        if "regime_id" in info and not done:
            regimes.append(int(info["regime_id"]))
    m = env.simulator.compute_metrics()
    out = {f: float(getattr(m, f)) for f in _METRIC_FIELDS}
    out.update(window_start=int(start), inventory_trajectory=inv,
               action_trajectory=actions, regime_trajectory=regimes,
               observation_trajectory=observations)
    return out


def rollout_baseline_window(name: str, df: pd.DataFrame, start: int,
                            horizon: int = HORIZON_STEPS, target_inventory: float = TARGET_INVENTORY,
                            side: str = SIDE, history_bars: int = 780) -> Dict[str, Any]:
    """Run TWAP / VWAP / POV on one window with only pre-window history available."""
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner

    strategy = {
        NAME_TWAP: lambda: TWAPStrategy(target_inventory=target_inventory, total_steps=horizon),
        NAME_VWAP: lambda: VWAPStrategy(target_inventory=target_inventory, total_steps=horizon),
        NAME_POV: lambda: POVStrategy(target_inventory=target_inventory, target_rate=0.10),
    }[name]()
    runner = BaselineRunner()
    window = df.iloc[start:start + horizon].reset_index(drop=True)
    history = df.iloc[max(0, start - history_bars):start]
    res = runner.run(strategy=strategy, market_data=window, target_inventory=target_inventory,
                     side=side, horizon_steps=horizon, history=history)
    inv = [float(target_inventory)] + [float(s.remaining_inventory) for s in runner.simulator.execution_history]
    return {
        "implementation_shortfall_bps": float(res.implementation_shortfall_bps),
        "implementation_shortfall": float(res.implementation_shortfall),
        "total_execution_cost": float(res.execution_cost),
        "total_impact_cost": float(res.market_impact_cost),
        "total_transaction_fees": float(res.total_transaction_fees),
        "terminal_penalty": float(res.terminal_penalty),
        "fill_rate": float(res.completion_rate),
        "vwap_slippage_bps": float(res.vwap_slippage_bps),
        "average_execution_price": float(res.average_execution_price),
        "arrival_price": float(res.arrival_price),
        "vwap_market_price": float(res.market_vwap_price),
        "executed_inventory": float(res.executed_inventory),
        "remaining_inventory": float(res.remaining_inventory),
        "window_start": int(start),
        "inventory_trajectory": inv,
        "action_trajectory": [],
        "regime_trajectory": [],
        "observation_trajectory": [],
    }


def evaluate_agent_windows(agent, env, starts: Sequence[int], name: str, seed: int = 0) -> pd.DataFrame:
    rows = []
    for w, s in enumerate(starts):
        r = rollout_agent_window(agent, env, s, seed=seed + w)
        rows.append({"strategy": name, "window": w, **{k: v for k, v in r.items() if not k.endswith("trajectory")}})
    return pd.DataFrame(rows)


def evaluate_baseline_windows(df: pd.DataFrame, starts: Sequence[int]) -> pd.DataFrame:
    rows = []
    for name in BASELINE_NAMES:
        for w, s in enumerate(starts):
            r = rollout_baseline_window(name, df, s)
            rows.append({"strategy": name, "window": w, **{k: v for k, v in r.items() if not k.endswith("trajectory")}})
    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def paired_stats(diff: Sequence[float], n_boot: int = 4000, seed: int = 0) -> Dict[str, float]:
    """
    Summary of a vector of paired differences (treatment minus control, in IS bps;
    negative = treatment is cheaper). Bootstrap 95% CI of the mean and a
    Wilcoxon signed-rank p-value.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "win_rate": float("nan")}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:  # pragma: no cover
        p = float("nan")
    return {"n": int(n), "mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "win_rate": float((d < 0).mean())}
