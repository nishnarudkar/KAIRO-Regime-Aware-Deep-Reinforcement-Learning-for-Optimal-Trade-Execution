"""
Abstract base class and standardized result container for all execution baselines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaselineResult:
    """
    Standardized execution performance report returned by every baseline runner.

    All monetary values are in the same currency as the market price data.
    All ratio / bps values are dimensionless or in basis points respectively.
    """
    strategy_name: str

    # --- Inventory ---
    target_inventory: float
    executed_inventory: float
    remaining_inventory: float
    completion_rate: float          # executed / target  [0, 1]

    # --- Prices ---
    arrival_price: float            # mid price at t=0
    average_execution_price: float  # VWAP of fills
    market_vwap_price: float        # VWAP of market mid prices over horizon

    # --- Cost Metrics ---
    implementation_shortfall: float      # $  IS vs arrival price
    implementation_shortfall_bps: float  # bps IS
    execution_cost: float                # $ total cost (IS + fees)
    market_impact_cost: float            # $ total temporary impact cost
    total_transaction_fees: float        # $ commission charges
    terminal_penalty: float              # $ penalty for unfilled inventory

    # --- Slippage ---
    vwap_slippage_bps: float            # bps vs market VWAP

    # --- Timing ---
    execution_duration_steps: int        # number of steps where fills occurred
    total_steps: int                     # total horizon steps

    # --- Market Turnover ---
    turnover: float                      # executed_inventory * average_execution_price  [$]


class BaseExecutionStrategy(ABC):
    """Abstract interface for deterministic execution baseline strategies."""

    @abstractmethod
    def get_action(
        self,
        remaining_inventory: float,
        time_step: int,
        total_steps: int,
        **kwargs,
    ) -> float:
        """
        Return the desired execution quantity at the current time step.

        Args:
            remaining_inventory: Shares not yet filled.
            time_step: 0-indexed current step.
            total_steps: Total steps in horizon.
            **kwargs: Strategy-specific extra context (e.g. current_volume).

        Returns:
            Requested quantity (>= 0); will be capped by the simulator.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Short human-readable strategy identifier."""
        ...
