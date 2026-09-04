"""
Gymnasium MDP Optimal Trade Execution Environment interfaces.
"""

from abc import ABC, abstractmethod
import numpy as np

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
