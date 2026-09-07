"""
Baseline Execution Strategy: TWAP (Time-Weighted Average Price).

Distributes the parent order uniformly over the execution horizon in equal
slices per time step.

Information available at each step t:
  - remaining_inventory  : shares still to execute
  - time_step            : current step index (0-indexed)
  - total_steps          : total number of steps in the horizon
  - steps_remaining      : how many steps (including current) are left

No market data (price, volume, spread) is used or required by TWAP.
This makes it inherently forward-looking-free.
"""

from src.baselines.base import BaseExecutionStrategy
import math


class TWAPStrategy(BaseExecutionStrategy):
    """
    Time-Weighted Average Price (TWAP) execution strategy.

    At each step the strategy targets an equal share of the total order:
        target_slice = target_inventory / total_steps

    To handle rounding and ensure full completion we floor each slice and
    flush the entire remaining inventory on the final step.
    """

    def __init__(self, target_inventory: float, total_steps: int):
        """
        Args:
            target_inventory: Total shares to execute over the horizon.
            total_steps: Number of time steps in the horizon.
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if target_inventory < 0:
            raise ValueError("target_inventory must be non-negative.")

        self.target_inventory = float(target_inventory)
        self.total_steps = int(total_steps)
        # Equal slice per step, floored to whole shares
        self._base_slice = math.floor(self.target_inventory / self.total_steps)
        # Any remainder is added to the last step
        self._remainder = self.target_inventory - self._base_slice * self.total_steps

    def get_action(
        self,
        remaining_inventory: float,
        time_step: int,
        total_steps: int,
        **kwargs,
    ) -> float:
        """
        Return the quantity to execute at this time step.

        Args:
            remaining_inventory: Shares not yet executed.
            time_step: 0-based index of the current step.
            total_steps: Total number of steps (must match constructor value).
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            Quantity to request at this step (>= 0).
        """
        # On the last step flush everything to guarantee full execution.
        if time_step >= total_steps - 1:
            return float(remaining_inventory)

        return float(min(self._base_slice, remaining_inventory))

    def name(self) -> str:
        return "TWAP"
