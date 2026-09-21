"""
Evaluation engine for Stage 6 DQN Agent.

Produces the same standardized metrics as BaselineResult so that
DQN results can be directly compared against TWAP / VWAP / POV.

Metrics (matching BaselineResult):
  - implementation_shortfall, implementation_shortfall_bps
  - execution_cost, market_impact_cost, total_transaction_fees
  - completion_rate
  - average_execution_price
  - vwap_slippage_bps
  - terminal_penalty
  - action distribution (proportion of each discrete action taken)
  - inventory trajectory (list of remaining_inventory at each step)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AgentEvalResult:
    """
    Standardized evaluation result for a DRL agent rollout.
    Mirrors BaselineResult fields exactly for fair comparison.
    """
    agent_name: str
    seed: int

    # Inventory
    target_inventory: float
    executed_inventory: float
    remaining_inventory: float
    completion_rate: float           # executed / target

    # Prices
    arrival_price: float
    average_execution_price: float
    market_vwap_price: float

    # Cost metrics
    implementation_shortfall: float       # $
    implementation_shortfall_bps: float   # bps
    execution_cost: float                 # $
    market_impact_cost: float             # $
    total_transaction_fees: float         # $
    terminal_penalty: float               # $

    # Slippage
    vwap_slippage_bps: float

    # Timing
    execution_duration_steps: int
    total_steps: int
    turnover: float

    # RL-specific extras
    action_counts: Dict[int, int]         # {action_idx: count}
    action_labels: Dict[int, str]         # {action_idx: "0%"|"10%"|"25%"|"50%"}
    reward_total: float
    inventory_trajectory: List[float]     # remaining_inventory at each step
    action_trajectory: List[int]          # action chosen at each step


def evaluate_agent(
    agent,
    market_data: pd.DataFrame,
    target_inventory: float = 100_000.0,
    side: str = "BUY",
    horizon_steps: int = 30,
    seed: int = 42,
    agent_name: str = "DQN",
    deterministic: bool = True,
    n_episodes: int = 1,
) -> AgentEvalResult:
    """
    Roll out a trained agent on evaluation market data and compute metrics.

    IMPORTANT: market_data must be the TEST split only (chronologically after train).
               No future information should appear in any observation.

    Args:
        agent: A trained DQNAgent (or any object with a .predict() method).
        market_data: Evaluation-split DataFrame.
        target_inventory: Total shares to execute.
        side: 'BUY' or 'SELL'.
        horizon_steps: Episode length in steps.
        seed: Seed for environment reset (determinism).
        agent_name: Label for the result object.
        deterministic: Use greedy policy (no exploration).
        n_episodes: Number of rollout episodes to average over.
                    Use >1 with stochastic policies only.

    Returns:
        AgentEvalResult with all execution metrics.
    """
    from src.environment import TradeExecutionEnv

    env = TradeExecutionEnv(
        market_data=market_data,
        target_inventory=target_inventory,
        side=side,
        horizon_steps=horizon_steps,
    )

    action_labels = {0: "0%", 1: "10%", 2: "25%", 3: "50%"}

    # Aggregate across episodes (typically n_episodes=1 for deterministic eval)
    all_results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        inv_traj: List[float] = [env.simulator.remaining_inventory]
        act_traj: List[int] = []
        action_counter: Counter = Counter()

        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            inv_traj.append(info["remaining_inventory"])
            act_traj.append(int(action))
            action_counter[int(action)] += 1

        # Final metrics from simulator
        metrics = env.simulator.compute_metrics()
        turnover = metrics.executed_inventory * metrics.average_execution_price

        all_results.append({
            "target_inventory": metrics.target_inventory,
            "executed_inventory": metrics.executed_inventory,
            "remaining_inventory": metrics.remaining_inventory,
            "completion_rate": metrics.fill_rate,
            "arrival_price": metrics.arrival_price,
            "average_execution_price": metrics.average_execution_price,
            "market_vwap_price": metrics.vwap_market_price,
            "implementation_shortfall": metrics.implementation_shortfall,
            "implementation_shortfall_bps": metrics.implementation_shortfall_bps,
            "execution_cost": metrics.total_execution_cost,
            "market_impact_cost": metrics.total_impact_cost,
            "total_transaction_fees": metrics.total_transaction_fees,
            "terminal_penalty": metrics.terminal_penalty,
            "vwap_slippage_bps": metrics.vwap_slippage_bps,
            "execution_duration_steps": len([a for a in act_traj if a > 0]),
            "total_steps": horizon_steps,
            "turnover": turnover,
            "reward_total": total_reward,
            "action_counts": dict(action_counter),
            "inventory_trajectory": inv_traj,
            "action_trajectory": act_traj,
        })

    # Average scalar fields; keep last episode's trajectories
    def _avg(key):
        vals = [r[key] for r in all_results if isinstance(r[key], (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    last = all_results[-1]

    return AgentEvalResult(
        agent_name=agent_name,
        seed=seed,
        target_inventory=_avg("target_inventory"),
        executed_inventory=_avg("executed_inventory"),
        remaining_inventory=_avg("remaining_inventory"),
        completion_rate=_avg("completion_rate"),
        arrival_price=_avg("arrival_price"),
        average_execution_price=_avg("average_execution_price"),
        market_vwap_price=_avg("market_vwap_price"),
        implementation_shortfall=_avg("implementation_shortfall"),
        implementation_shortfall_bps=_avg("implementation_shortfall_bps"),
        execution_cost=_avg("execution_cost"),
        market_impact_cost=_avg("market_impact_cost"),
        total_transaction_fees=_avg("total_transaction_fees"),
        terminal_penalty=_avg("terminal_penalty"),
        vwap_slippage_bps=_avg("vwap_slippage_bps"),
        execution_duration_steps=int(_avg("execution_duration_steps")),
        total_steps=int(_avg("total_steps")),
        turnover=_avg("turnover"),
        reward_total=_avg("reward_total"),
        action_counts=last["action_counts"],
        action_labels=action_labels,
        inventory_trajectory=last["inventory_trajectory"],
        action_trajectory=last["action_trajectory"],
    )
