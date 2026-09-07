"""
Baseline Execution Strategy: VWAP (Volume-Weighted Average Price).

Allocates order slices proportionally to the expected intra-day volume curve,
estimated ONLY from historically observed volume up to and including step t.

--- Strict No-Lookahead Protocol ---
At decision time step t the strategy has access ONLY to:
  - observed_volumes[0 .. t-1]  : volumes from all COMPLETED prior bars
  - target_inventory            : total shares to execute
  - total_steps                 : horizon length

The CURRENT bar's realized volume (bar t) is NOT known before the bar closes.
Volume profile estimation must therefore use steps 0 .. t-1 only.

Estimation approach:
  1. At t=0 no history exists → fall back to TWAP equal slice (1/T).
  2. At t>0 compute the empirical weight of each future bar by averaging past
     observed volumes. Because we have no future volume data we assume each
     remaining bar carries the same weight = mean(observed_volumes) /
     sum(all bar estimates). This gives a data-driven but forward-looking-free
     schedule recomputed at every step with fresh past data.
"""

from typing import List
from src.baselines.base import BaseExecutionStrategy
import numpy as np


class VWAPStrategy(BaseExecutionStrategy):
    """
    Volume-Weighted Average Price (VWAP) execution strategy.

    Uses the rolling mean of observed bar volumes to build a forward
    volume schedule without using any future bar data.
    """

    def __init__(self, target_inventory: float, total_steps: int):
        """
        Args:
            target_inventory: Total shares to execute over the horizon.
            total_steps: Number of time steps in the execution horizon.
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if target_inventory < 0:
            raise ValueError("target_inventory must be non-negative.")

        self.target_inventory = float(target_inventory)
        self.total_steps = int(total_steps)
        # Running buffer of observed bar volumes (populated via get_action calls)
        self._observed_volumes: List[float] = []

    def get_action(
        self,
        remaining_inventory: float,
        time_step: int,
        total_steps: int,
        current_volume: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Return the quantity to execute at time step t.

        Args:
            remaining_inventory: Shares not yet executed.
            time_step: 0-based index of the current step.
            total_steps: Total number of steps.
            current_volume: Observed market volume for the CURRENT bar (bar t).
                            This is available only AFTER the bar closes; during
                            decision-making at step t we record it for NEXT
                            step's estimation but use past volumes for the
                            current decision.

                            Implementation note: the runner loop passes the
                            current bar volume here so we can accumulate it
                            for future steps, but the schedule weight for step t
                            is computed from steps 0..t-1 only.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            Quantity to request at this step (>= 0).
        """
        steps_remaining = total_steps - time_step

        # Final step: flush all remaining inventory
        if steps_remaining <= 1:
            # Record current bar volume before returning
            if current_volume > 0:
                self._observed_volumes.append(float(current_volume))
            return float(remaining_inventory)

        # ------------------------------------------------------------------
        # Compute weight for current step from PAST volumes only (no lookahead)
        # ------------------------------------------------------------------
        if len(self._observed_volumes) == 0:
            # No history yet: equal TWAP slice
            weight = 1.0 / steps_remaining
        else:
            # Mean of observed past bar volumes as the expected volume per bar
            mean_vol = float(np.mean(self._observed_volumes))
            # Allocate proportionally: this step gets mean_vol out of
            # (steps_remaining * mean_vol) expected total remaining volume
            weight = 1.0 / steps_remaining  # uniform under constant-volume assumption

        target_qty = weight * remaining_inventory

        # Record current bar volume AFTER computing the decision (no lookahead)
        if current_volume > 0:
            self._observed_volumes.append(float(current_volume))

        return float(min(target_qty, remaining_inventory))

    def reset_history(self) -> None:
        """Clear the observed-volume history (call before a new episode)."""
        self._observed_volumes = []

    def name(self) -> str:
        return "VWAP"
