"""
ExecutionSimulator Engine for Large-Order Trade Execution.

Tracks market replay state, inventory, transaction costs, market impact,
slippage, and execution performance metrics without lookahead bias.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.execution.impact_models import BaseImpactModel, LinearImpactModel, AlmgrenChrissImpactModel


@dataclass
class StepResult:
    """Execution outcome for a single simulation timestep."""
    step: int
    timestamp: pd.Timestamp
    requested_qty: float
    filled_qty: float
    unfilled_qty: float
    remaining_inventory: float
    market_mid_price: float
    bid_price: float
    ask_price: float
    spread: float
    execution_price: float
    temporary_impact: float
    permanent_impact: float
    spread_cost: float
    transaction_fee: float
    step_cost: float
    is_done: bool


@dataclass
class ExecutionSummary:
    """Aggregated execution metrics across the full order execution horizon."""
    target_inventory: float
    executed_inventory: float
    remaining_inventory: float
    fill_rate: float
    arrival_price: float
    average_execution_price: float
    vwap_market_price: float
    implementation_shortfall: float
    implementation_shortfall_bps: float
    vwap_slippage_bps: float
    total_execution_cost: float
    total_impact_cost: float
    total_transaction_fees: float
    terminal_penalty: float
    total_steps: int
    side: str


class ExecutionSimulator:
    """
    Market replay execution simulator for large parent orders.
    
    Ensures strict temporal causality (no lookahead data leakage) and
    segregates exogenous market observations from endogenously simulated outcomes.
    """

    def __init__(
        self,
        impact_model: Optional[BaseImpactModel] = None,
        max_participation_rate: float = 0.15,
        per_share_fee: float = 0.0005,
        per_trade_fee: float = 0.0,
        bps_fee: float = 0.0,
        default_spread_bps: float = 2.0,
        terminal_penalty_mult: float = 2.0,
    ):
        """
        Args:
            impact_model: Market impact model instance. Defaults to LinearImpactModel().
            max_participation_rate: Maximum fraction of bar volume allowed to fill per step (e.g. 0.15 = 15%).
            per_share_fee: Fixed commission per executed share ($/share).
            per_trade_fee: Fixed commission per order submission ($/trade).
            bps_fee: Basis point commission fee (1 bps = 0.0001).
            default_spread_bps: Default spread in bps if bid/ask not present in market data.
            terminal_penalty_mult: Penalty multiplier applied to temporary impact for unexecuted inventory at horizon end.
        """
        self.impact_model = impact_model if impact_model is not None else LinearImpactModel()
        self.max_participation_rate = max_participation_rate
        self.per_share_fee = per_share_fee
        self.per_trade_fee = per_trade_fee
        self.bps_fee = bps_fee
        self.default_spread_bps = default_spread_bps
        self.terminal_penalty_mult = terminal_penalty_mult

        # State initialization
        self.market_data: Optional[pd.DataFrame] = None
        self.target_inventory: float = 0.0
        self.side: str = "BUY"
        self.horizon_steps: int = 0
        self.current_step: int = 0
        
        self.remaining_inventory: float = 0.0
        self.executed_inventory: float = 0.0
        self.arrival_price: float = 0.0
        
        self.accumulated_permanent_impact: float = 0.0
        self.execution_history: List[StepResult] = []

    def reset(
        self,
        market_data: pd.DataFrame,
        target_inventory: float = 100000.0,
        side: str = "BUY",
        horizon_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reset simulation state with new market replay data and order parameters.
        
        Args:
            market_data: DataFrame with market columns: ['timestamp', 'price'/'close', 'volume']
                         and optionally ['bid', 'ask', 'spread', 'volatility'].
            target_inventory: Target order volume in shares (e.g., 100,000).
            side: 'BUY' or 'SELL'.
            horizon_steps: Number of steps for execution horizon. Defaults to len(market_data).
            
        Returns:
            Initial state snapshot dictionary at step 0.
        """
        if market_data.empty:
            raise ValueError("market_data DataFrame cannot be empty.")
        
        # Standardize market data columns
        df = market_data.copy()
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            else:
                df['timestamp'] = pd.date_range(start="2026-01-01 09:30", periods=len(df), freq="1min")

        if 'price' not in df.columns:
            if 'close' in df.columns:
                df['price'] = df['close']
            else:
                raise KeyError("market_data must contain 'price' or 'close' column.")

        if 'volume' not in df.columns:
            raise KeyError("market_data must contain 'volume' column.")

        if 'volatility' not in df.columns:
            # Estimate volatility via rolling std of returns if not provided
            returns = df['price'].pct_change().fillna(0.0)
            df['volatility'] = returns.rolling(window=10, min_periods=1).std().fillna(0.001)

        self.market_data = df.reset_index(drop=True)
        self.target_inventory = float(target_inventory)
        self.side = side.upper()
        if self.side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")

        total_bars = len(self.market_data)
        self.horizon_steps = horizon_steps if horizon_steps is not None else total_bars
        self.horizon_steps = min(self.horizon_steps, total_bars)

        self.current_step = 0
        self.remaining_inventory = self.target_inventory
        self.executed_inventory = 0.0
        self.arrival_price = float(self.market_data.loc[0, 'price'])
        self.accumulated_permanent_impact = 0.0
        self.execution_history = []

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """
        Get current observable state and simulator execution progress.
        STRICT TEMPORAL ISOLATION: Only accesses market_data at self.current_step.
        """
        if self.market_data is None:
            raise RuntimeError("Simulator has not been initialized. Call reset() first.")

        # Ensure current step doesn't exceed market data bounds
        step_idx = min(self.current_step, len(self.market_data) - 1)
        row = self.market_data.iloc[step_idx]

        timestamp = row['timestamp']
        base_mid_price = float(row['price'])
        
        # Apply accumulated permanent impact from past execution steps to mid price
        current_mid_price = base_mid_price + self.accumulated_permanent_impact

        # Determine Bid / Ask quotes
        if 'bid' in row and 'ask' in row and pd.notna(row['bid']) and pd.notna(row['ask']):
            bid = float(row['bid']) + self.accumulated_permanent_impact
            ask = float(row['ask']) + self.accumulated_permanent_impact
            spread = ask - bid
        else:
            spread = current_mid_price * (self.default_spread_bps / 10000.0)
            half_spread = spread / 2.0
            bid = current_mid_price - half_spread
            ask = current_mid_price + half_spread

        volume = float(row['volume'])
        volatility = float(row['volatility'])

        is_done = (self.current_step >= self.horizon_steps) or (self.remaining_inventory <= 0.0)

        # Average execution price so far
        if self.executed_inventory > 0 and len(self.execution_history) > 0:
            total_spent = sum(item.filled_qty * item.execution_price for item in self.execution_history)
            avg_exec_price = total_spent / self.executed_inventory
        else:
            avg_exec_price = 0.0

        return {
            "current_step": self.current_step,
            "current_timestamp": timestamp,
            "price": current_mid_price,
            "current_price": current_mid_price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "volume": volume,
            "volatility": volatility,
            "target_inventory": self.target_inventory,
            "remaining_inventory": self.remaining_inventory,
            "executed_inventory": self.executed_inventory,
            "average_execution_price": avg_exec_price,
            "elapsed_time": self.current_step,
            "remaining_time": max(0, self.horizon_steps - self.current_step),
            "is_done": is_done,
            "side": self.side
        }

    def step(self, order_qty: float) -> StepResult:
        """
        Execute a market order for step $t$, advance simulation clock by 1 minute.
        
        Args:
            order_qty: Requested execution volume in shares for this timestep.
            
        Returns:
            StepResult object detailing execution filled qty, prices, impact, fees, and state.
        """
        if self.market_data is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")
        if self.is_done():
            raise RuntimeError("Simulation has completed. Call reset() to start a new run.")

        state = self.get_state()
        step_idx = self.current_step
        
        # 1. Clamp order quantity to prevent over-execution or negative inventory
        requested_qty = float(max(0.0, order_qty))
        qty_to_fill = min(requested_qty, self.remaining_inventory)

        # 2. Apply participation rate cap (partial fill rule)
        max_fillable_volume = max(0.0, state['volume'] * self.max_participation_rate)
        filled_qty = min(qty_to_fill, max_fillable_volume)
        unfilled_qty = requested_qty - filled_qty

        mid_price = state['price']
        bid_price = state['bid']
        ask_price = state['ask']
        spread = state['spread']
        volatility = state['volatility']

        if filled_qty > 0:
            # 3. Calculate market impact
            temp_impact = self.impact_model.calculate_impact(
                trade_size=filled_qty,
                volume=state['volume'],
                volatility=volatility,
                price=mid_price,
                side=self.side
            )

            perm_impact = self.impact_model.calculate_permanent_impact(
                trade_size=filled_qty,
                volume=state['volume'],
                price=mid_price,
                side=self.side
            )

            # 4. Calculate Spread & Execution Base Price
            base_start_price = ask_price if self.side == "BUY" else bid_price
            spread_cost = abs(base_start_price - mid_price)

            # 5. Total Effective Execution Price
            execution_price = base_start_price + temp_impact

            # 6. Transaction Fees
            transaction_fee = (
                self.per_trade_fee +
                self.per_share_fee * filled_qty +
                self.bps_fee * 0.0001 * (filled_qty * execution_price)
            )

            step_cost = (filled_qty * execution_price) + transaction_fee
        else:
            temp_impact = 0.0
            perm_impact = 0.0
            spread_cost = 0.0
            execution_price = ask_price if self.side == "BUY" else bid_price
            transaction_fee = 0.0
            step_cost = 0.0

        # Update Inventory Accounting
        self.remaining_inventory -= filled_qty
        self.executed_inventory += filled_qty
        
        # Accumulate permanent impact for subsequent steps
        self.accumulated_permanent_impact += perm_impact

        # Advance Time Step
        self.current_step += 1
        is_done = (self.current_step >= self.horizon_steps) or (self.remaining_inventory <= 0.0)

        result = StepResult(
            step=step_idx,
            timestamp=state['current_timestamp'],
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            unfilled_qty=unfilled_qty,
            remaining_inventory=self.remaining_inventory,
            market_mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            execution_price=execution_price,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            spread_cost=spread_cost,
            transaction_fee=transaction_fee,
            step_cost=step_cost,
            is_done=is_done
        )

        self.execution_history.append(result)
        return result

    def is_done(self) -> bool:
        """Check if simulation horizon is reached or inventory fully executed."""
        if self.market_data is None:
            return True
        return (self.current_step >= self.horizon_steps) or (self.remaining_inventory <= 0.0)

    def compute_metrics(self) -> ExecutionSummary:
        """
        Compute total summary metrics: Implementation Shortfall, VWAP benchmark,
        fees, total costs, and terminal penalties.
        """
        if not self.execution_history:
            return ExecutionSummary(
                target_inventory=self.target_inventory,
                executed_inventory=0.0,
                remaining_inventory=self.target_inventory,
                fill_rate=0.0,
                arrival_price=self.arrival_price,
                average_execution_price=0.0,
                vwap_market_price=self.arrival_price,
                implementation_shortfall=0.0,
                implementation_shortfall_bps=0.0,
                vwap_slippage_bps=0.0,
                total_execution_cost=0.0,
                total_impact_cost=0.0,
                total_transaction_fees=0.0,
                terminal_penalty=0.0,
                total_steps=0,
                side=self.side
            )

        total_executed = sum(res.filled_qty for res in self.execution_history)
        fill_rate = total_executed / self.target_inventory if self.target_inventory > 0 else 0.0

        if total_executed > 0:
            avg_exec_price = sum(res.filled_qty * res.execution_price for res in self.execution_history) / total_executed
        else:
            avg_exec_price = 0.0

        # VWAP Market Price over the steps executed/horizon
        executed_data = self.market_data.iloc[:self.current_step]
        vwap_market_price = float((executed_data['price'] * executed_data['volume']).sum() / executed_data['volume'].sum())

        # Implementation Shortfall relative to arrival mid price
        sign = 1.0 if self.side == "BUY" else -1.0
        
        # IS for executed portion
        executed_is = sign * sum(res.filled_qty * (res.execution_price - self.arrival_price) for res in self.execution_history)

        # Terminal Incomplete-Order Penalty if remaining inventory > 0
        terminal_penalty = 0.0
        if self.remaining_inventory > 0:
            last_row = self.market_data.iloc[min(self.current_step - 1, len(self.market_data) - 1)]
            last_price = float(last_row['price'])
            last_vol = float(last_row['volume'])
            last_vola = float(last_row['volatility'])
            
            # Heavy liquidation impact for leftover inventory
            penalty_impact = self.impact_model.calculate_impact(
                trade_size=self.remaining_inventory,
                volume=last_vol,
                volatility=last_vola,
                price=last_price,
                side=self.side
            ) * self.terminal_penalty_mult

            penalty_price = (last_price + (last_row.get('spread', last_price * 0.0002) / 2.0)) + penalty_impact
            terminal_penalty = self.remaining_inventory * abs(penalty_price - self.arrival_price)

        total_is = executed_is + terminal_penalty
        is_bps = (total_is / (self.target_inventory * self.arrival_price)) * 10000.0 if self.target_inventory > 0 else 0.0

        # VWAP Slippage in bps
        if avg_exec_price > 0 and vwap_market_price > 0:
            vwap_slippage_bps = sign * ((avg_exec_price - vwap_market_price) / vwap_market_price) * 10000.0
        else:
            vwap_slippage_bps = 0.0

        total_impact_cost = sum(res.filled_qty * abs(res.temporary_impact) for res in self.execution_history)
        total_fees = sum(res.transaction_fee for res in self.execution_history)
        total_exec_cost = total_is + total_fees

        return ExecutionSummary(
            target_inventory=self.target_inventory,
            executed_inventory=total_executed,
            remaining_inventory=self.remaining_inventory,
            fill_rate=fill_rate,
            arrival_price=self.arrival_price,
            average_execution_price=avg_exec_price,
            vwap_market_price=vwap_market_price,
            implementation_shortfall=total_is,
            implementation_shortfall_bps=is_bps,
            vwap_slippage_bps=vwap_slippage_bps,
            total_execution_cost=total_exec_cost,
            total_impact_cost=total_impact_cost,
            total_transaction_fees=total_fees,
            terminal_penalty=terminal_penalty,
            total_steps=len(self.execution_history),
            side=self.side
        )
