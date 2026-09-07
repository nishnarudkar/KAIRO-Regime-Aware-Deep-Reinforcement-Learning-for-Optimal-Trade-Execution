"""
Baseline Execution Strategy: POV (Percentage of Volume).

Executes a configurable fraction of the market volume observed in each bar.

--- Strict No-Lookahead Protocol ---
At decision time step t the strategy has access ONLY to:
  - current_volume  : volume observed in the current bar (bar t)
                      This is the bar that is CLOSING now. The trade is
                      submitted as a market order executing into this bar's
                      liquidity, so using bar-t volume for the decision is
                      realistic (this is how POV desks operate in practice:
                      they see real-time volume prints and target X% of it).
  - remaining_inventory : shares still to execute
  - target_rate     : configured participation rate (e.g. 0.10 = 10%)

No future bar data is accessed.
"""

from src.baselines.base import BaseExecutionStrategy


class POVStrategy(BaseExecutionStrategy):
    """
    Percentage-of-Volume (POV) execution strategy.

    At each step the strategy targets:
        order_qty = min(target_rate * current_volume, remaining_inventory)

    The execution simulator then further caps filled_qty by its own
    max_participation_rate, so the realized fill may be less than requested.
    """

    def __init__(
        self,
        target_inventory: float,
        target_rate: float = 0.10,
    ):
        """
        Args:
            target_inventory: Total shares to execute over the horizon.
            target_rate: Fraction of each bar's volume to target (0, 1].
                         E.g. 0.10 means "execute 10% of observed market volume".
        """
        if not (0 < target_rate <= 1.0):
            raise ValueError("target_rate must be in (0, 1].")
        if target_inventory < 0:
            raise ValueError("target_inventory must be non-negative.")

        self.target_inventory = float(target_inventory)
        self.target_rate = float(target_rate)

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
            total_steps: Total number of steps in the horizon.
            current_volume: Bar-t market volume (current bar closing volume).
                            Available at decision time — POV desks observe
                            real-time prints intra-bar and submit at close.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            Quantity to request at this step (>= 0).
        """
        if current_volume <= 0:
            return 0.0

        target_qty = self.target_rate * current_volume
        return float(min(target_qty, remaining_inventory))

    def name(self) -> str:
        return f"POV({self.target_rate:.0%})"
