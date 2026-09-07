"""
Gymnasium MDP Optimal Trade Execution Environment interfaces.
"""

from abc import ABC, abstractmethod

class BaseRewardCalculator(ABC):
    """Abstract interface for execution penalty and reward functions."""

    @abstractmethod
    def calculate_reward(
        self,
        execution_price: float,
        benchmark_price: float,
        executed_qty: float,
        remaining_qty: float,
        step: int,
        max_steps: int
    ) -> float:
        """Compute execution reward penalizing shortfall, impact, and risk."""
        pass


from src.environment.rewards import ModularExecutionReward
from src.environment.env import TradeExecutionEnv

__all__ = [
    "BaseRewardCalculator",
    "ModularExecutionReward",
    "TradeExecutionEnv",
]
