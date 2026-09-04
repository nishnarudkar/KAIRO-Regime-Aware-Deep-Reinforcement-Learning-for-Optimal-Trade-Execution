"""
Baseline execution strategies (TWAP, VWAP, POV).
"""

from abc import ABC, abstractmethod

class BaseExecutionStrategy(ABC):
    """Abstract interface for baseline deterministic execution algorithms."""

    @abstractmethod
    def get_action(self, remaining_inventory: float, time_step: int, total_steps: int) -> float:
        """Determine execution quantity for the current time step."""
        pass
