"""
Baseline Execution Strategy: VWAP (Volume-Weighted Average Price).

Allocates the parent order in proportion to an *ex-ante* intraday volume
profile, estimated only from bars that precede the execution window (average
volume by minute-of-day over earlier trading days).

--- Strict No-Lookahead Protocol ---
  - The profile is built from history strictly before the window start.
  - No volume of the current or future bars is used to size a slice.
  - If no history is available the profile is flat, which makes the strategy
    identical to TWAP. This is reported honestly through ``uses_profile``.
"""

from typing import List, Optional
from src.baselines.base import BaseExecutionStrategy
import numpy as np
import pandas as pd


def estimate_volume_profile(history: Optional[pd.DataFrame], window: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Expected relative volume for each bar of ``window``, from ``history`` only.

    Averages history volume by minute-of-day. Returns None when there is not
    enough history (fewer than one full pass over the window's minutes).
    """
    if history is None or len(history) < len(window) or "timestamp" not in history.columns:
        return None
    ts_h = pd.to_datetime(history["timestamp"])
    minute_h = (ts_h.dt.hour * 60 + ts_h.dt.minute).to_numpy()
    by_minute = pd.Series(history["volume"].to_numpy()).groupby(minute_h).mean()
    ts_w = pd.to_datetime(window["timestamp"])
    minute_w = (ts_w.dt.hour * 60 + ts_w.dt.minute).to_numpy()
    fallback = float(np.mean(history["volume"]))
    profile = np.array([float(by_minute.get(m, fallback)) for m in minute_w])
    if not np.all(np.isfinite(profile)) or profile.sum() <= 0:
        return None
    return profile


class VWAPStrategy(BaseExecutionStrategy):
    """
    Volume-Weighted Average Price (VWAP) execution strategy.

    Follows an ex-ante intraday volume profile estimated from history
    that precedes the execution window.
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
        # Optional ex-ante volume profile (length >= total_steps); None => flat (TWAP-like).
        self.volume_profile: Optional[np.ndarray] = None
        self._observed_volumes: List[float] = []

    @property
    def uses_profile(self) -> bool:
        return self.volume_profile is not None

    def set_volume_profile(self, profile: Optional[np.ndarray]) -> None:
        """Provide the ex-ante volume profile estimated from pre-window history."""
        self.volume_profile = None if profile is None else np.asarray(profile, dtype=float)

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

        # Record the current bar volume for diagnostics only (never used for sizing).
        if current_volume > 0:
            self._observed_volumes.append(float(current_volume))

        # Final step: flush all remaining inventory
        if steps_remaining <= 1:
            return float(remaining_inventory)

        profile = self.volume_profile
        if profile is not None and len(profile) >= total_steps and profile[time_step:total_steps].sum() > 0:
            weight = float(profile[time_step] / profile[time_step:total_steps].sum())
        else:
            weight = 1.0 / steps_remaining   # flat profile == TWAP

        return float(min(weight * remaining_inventory, remaining_inventory))

    def reset_history(self) -> None:
        """Clear the observed-volume history (call before a new episode)."""
        self._observed_volumes = []

    def name(self) -> str:
        return "VWAP"
