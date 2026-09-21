"""
Standardized Baseline Runner.

Executes any BaseExecutionStrategy through the shared ExecutionSimulator so
that TWAP, VWAP, POV, and future RL strategies all use:
  - the same initial order
  - the same market data
  - the same transaction-cost model
  - the same market-impact model
  - the same evaluation interface

Returns a BaselineResult with a standardised set of comparable metrics.
"""

from typing import Optional
import pandas as pd

from src.execution.simulator import ExecutionSimulator
from src.execution.impact_models import BaseImpactModel, default_impact_model
from src.baselines.base import BaseExecutionStrategy, BaselineResult


class BaselineRunner:
    """
    Runs a baseline strategy through the ExecutionSimulator and collects
    standardized execution metrics.

    The runner is responsible for:
      1. Resetting the simulator with the given market data and order params.
      2. Calling strategy.get_action() at each step with ONLY causally
         admissible information.
      3. Forwarding the action to simulator.step().
      4. Assembling a BaselineResult from simulator.compute_metrics().
    """

    def __init__(
        self,
        impact_model: Optional[BaseImpactModel] = None,
        max_participation_rate: float = 0.15,
        per_share_fee: float = 0.0005,
        default_spread_bps: float = 2.0,
    ):
        """
        Args:
            impact_model: Shared market-impact model instance.
            max_participation_rate: Max fraction of bar volume fillable per step.
            per_share_fee: Commission per executed share.
            default_spread_bps: Synthetic spread when bid/ask absent.
        """
        self.impact_model = impact_model or default_impact_model()
        self.simulator = ExecutionSimulator(
            impact_model=self.impact_model,
            max_participation_rate=max_participation_rate,
            per_share_fee=per_share_fee,
            default_spread_bps=default_spread_bps,
        )

    def run(
        self,
        strategy: BaseExecutionStrategy,
        market_data: pd.DataFrame,
        target_inventory: float = 100_000.0,
        side: str = "BUY",
        horizon_steps: Optional[int] = None,
        history: Optional[pd.DataFrame] = None,
    ) -> BaselineResult:
        """
        Execute strategy over market_data and return standardized metrics.

        Causal information contract per step t:
          ┌─────────────────────────────────────────────────────────┐
          │  Available at decision time (step t)                    │
          │  ─────────────────────────────────                      │
          │  • remaining_inventory          (endogenous state)      │
          │  • time_step, total_steps       (time index)            │
          │  • current bar mid/bid/ask/vol  (bar t closing data)    │
          │    → POV uses current_volume for its participation calc │
          │    → VWAP records bar-t volume AFTER making decision    │
          │    → TWAP ignores market data entirely                  │
          │  NOT available:                                         │
          │  • bar t+1, t+2, … (future bars)                       │
          └─────────────────────────────────────────────────────────┘

        Args:
            strategy: A BaseExecutionStrategy instance.
            market_data: DataFrame (timestamp, price/close, volume, …).
            target_inventory: Total shares to execute.
            side: 'BUY' or 'SELL'.
            horizon_steps: Number of steps; defaults to len(market_data).
            history: Optional bars preceding market_data (past only), used by VWAP
                     to estimate its volume profile.

        Returns:
            BaselineResult with all standardized metrics.
        """
        n = len(market_data)
        total_steps = min(horizon_steps if horizon_steps is not None else n, n)

        # VWAP needs an ex-ante volume profile; build it from pre-window history only.
        if hasattr(strategy, "set_volume_profile"):
            from src.baselines.vwap import estimate_volume_profile
            strategy.set_volume_profile(estimate_volume_profile(history, market_data.iloc[:total_steps]))

        self.simulator.reset(
            market_data=market_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=total_steps,
        )

        active_steps = 0  # steps in which at least one share was filled

        while not self.simulator.is_done():
            t = self.simulator.current_step
            state = self.simulator.get_state()

            # Build causal kwargs: pass only information from bar t
            action_kwargs = {
                "current_volume": state["volume"],  # bar-t volume (causal for POV/VWAP)
            }

            order_qty = strategy.get_action(
                remaining_inventory=state["remaining_inventory"],
                time_step=t,
                total_steps=total_steps,
                **action_kwargs,
            )

            result = self.simulator.step(order_qty=order_qty)
            if result.filled_qty > 0:
                active_steps += 1

        summary = self.simulator.compute_metrics()

        turnover = summary.executed_inventory * summary.average_execution_price

        return BaselineResult(
            strategy_name=strategy.name(),
            target_inventory=summary.target_inventory,
            executed_inventory=summary.executed_inventory,
            remaining_inventory=summary.remaining_inventory,
            completion_rate=summary.fill_rate,
            arrival_price=summary.arrival_price,
            average_execution_price=summary.average_execution_price,
            market_vwap_price=summary.vwap_market_price,
            implementation_shortfall=summary.implementation_shortfall,
            implementation_shortfall_bps=summary.implementation_shortfall_bps,
            execution_cost=summary.total_execution_cost,
            market_impact_cost=summary.total_impact_cost,
            total_transaction_fees=summary.total_transaction_fees,
            terminal_penalty=summary.terminal_penalty,
            vwap_slippage_bps=summary.vwap_slippage_bps,
            execution_duration_steps=active_steps,
            total_steps=total_steps,
            turnover=turnover,
        )
