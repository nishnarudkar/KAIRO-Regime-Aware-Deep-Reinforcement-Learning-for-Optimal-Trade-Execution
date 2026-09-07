"""
Example script demonstrating trade execution simulation.

Scenario:
- Stock: AAPL
- Side: BUY
- Target Order: 100,000 shares
- Horizon: 30 minutes (30 timesteps)
- Strategy: TWAP (Time-Weighted Average Price) baseline schedule
"""

import os
import pandas as pd

from src.execution import (
    ExecutionSimulator,
    AlmgrenChrissImpactModel,
    LinearImpactModel
)

def run_execution_demo():
    print("=" * 70)
    print("      TRADE EXECUTION SIMULATOR - REPLAY DEMO")
    print("=" * 70)

    # 1. Load deterministic market data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")
    market_df = pd.read_csv(data_path)
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])

    # 2. Configure Impact Model & Simulator
    impact_model = AlmgrenChrissImpactModel(eta=0.05, gamma=0.01, alpha=0.5)
    simulator = ExecutionSimulator(
        impact_model=impact_model,
        max_participation_rate=0.20,  # Max 20% bar volume fill rate
        per_share_fee=0.0005,
        default_spread_bps=2.0
    )

    # 3. Reset Simulation for Initial Scenario
    target_inventory = 100000.0
    horizon_steps = len(market_df)  # 30 steps
    side = "BUY"

    initial_state = simulator.reset(
        market_data=market_df,
        target_inventory=target_inventory,
        side=side,
        horizon_steps=horizon_steps
    )

    print(f"\n[INITIAL STATE]")
    print(f"  Asset: AAPL | Side: {side}")
    print(f"  Target Order: {initial_state['target_inventory']:,.0f} shares")
    print(f"  Horizon: {initial_state['remaining_time']} minutes (1-min steps)")
    print(f"  Arrival Mid Price: ${initial_state['current_price']:.2f}")
    print(f"  Bid: ${initial_state['bid']:.2f} | Ask: ${initial_state['ask']:.2f} (Spread: ${initial_state['spread']:.4f})")

    # 4. Define Execution Schedule (TWAP: ~3,333.33 shares per minute)
    twap_order_size = target_inventory / horizon_steps

    print("\n" + "-" * 70)
    print(f"{'Step':<5} | {'Timestamp':<19} | {'Req Qty':<9} | {'Filled':<9} | {'Remaining':<10} | {'Exec Price':<10} | {'Temp Impact':<11}")
    print("-" * 70)

    step_count = 0
    while not simulator.is_done():
        # Execute order for current timestep
        res = simulator.step(order_qty=twap_order_size)
        step_count += 1

        print(
            f"{res.step:<5} | "
            f"{str(res.timestamp):<19} | "
            f"{res.requested_qty:>9.0f} | "
            f"{res.filled_qty:>9.0f} | "
            f"{res.remaining_inventory:>10.0f} | "
            f"${res.execution_price:>9.2f} | "
            f"${res.temporary_impact:>10.4f}"
        )

    # 5. Compute Aggregate Execution Performance Metrics
    metrics = simulator.compute_metrics()

    print("=" * 70)
    print("      SIMULATION COMPLETED - FINAL EXECUTION METRICS")
    print("=" * 70)
    print(f"  Total Horizon Steps:            {metrics.total_steps}")
    print(f"  Target Inventory:               {metrics.target_inventory:,.0f} shares")
    print(f"  Executed Inventory:             {metrics.executed_inventory:,.0f} shares ({metrics.fill_rate*100:.1f}%)")
    print(f"  Remaining Inventory:            {metrics.remaining_inventory:,.0f} shares")
    print(f"  Arrival Mid Price:              ${metrics.arrival_price:.2f}")
    print(f"  Average Execution Price:        ${metrics.average_execution_price:.4f}")
    print(f"  Market VWAP Price:              ${metrics.vwap_market_price:.4f}")
    print(f"  Implementation Shortfall ($):    ${metrics.implementation_shortfall:,.2f}")
    print(f"  Implementation Shortfall (bps):  {metrics.implementation_shortfall_bps:.2f} bps")
    print(f"  VWAP Slippage (bps):            {metrics.vwap_slippage_bps:.2f} bps")
    print(f"  Total Market Impact Cost ($):   ${metrics.total_impact_cost:,.2f}")
    print(f"  Total Transaction Fees ($):     ${metrics.total_transaction_fees:,.2f}")
    print(f"  Terminal Unfilled Penalty ($):  ${metrics.terminal_penalty:,.2f}")
    print(f"  Total Execution Cost ($):       ${metrics.total_execution_cost:,.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_execution_demo()
