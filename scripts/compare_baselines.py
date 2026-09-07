"""
Baseline Strategy Comparison Script.

Runs TWAP, VWAP, and POV through the shared ExecutionSimulator on the
deterministic AAPL test dataset and prints a side-by-side metrics table.

Usage:
    python -m scripts.compare_baselines
"""

import os
import sys
import pandas as pd

from src.baselines import (
    TWAPStrategy,
    VWAPStrategy,
    POVStrategy,
    BaselineRunner,
    BaselineResult,
)
from src.execution.impact_models import AlmgrenChrissImpactModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")
TARGET_INVENTORY = 100_000.0
HORIZON_STEPS    = 30
SIDE             = "BUY"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def print_result(result: BaselineResult) -> None:
    print(f"\n{'-'*60}")
    print(f"  Strategy : {result.strategy_name}")
    print(f"{'-'*60}")
    print(f"  Arrival Price              : ${result.arrival_price:.4f}")
    print(f"  Avg Execution Price        : ${result.average_execution_price:.4f}")
    print(f"  Market VWAP Price          : ${result.market_vwap_price:.4f}")
    print(f"  Target Inventory           : {result.target_inventory:>12,.0f} shares")
    print(f"  Executed Inventory         : {result.executed_inventory:>12,.0f} shares")
    print(f"  Remaining Inventory        : {result.remaining_inventory:>12,.0f} shares")
    print(f"  Completion Rate            : {result.completion_rate*100:>10.1f} %")
    print(f"  Execution Duration         : {result.execution_duration_steps:>10} / {result.total_steps} steps")
    print(f"  Turnover                   : ${result.turnover:>14,.2f}")
    print(f"  Implementation Shortfall   : ${result.implementation_shortfall:>14,.2f}")
    print(f"  IS (bps)                   : {result.implementation_shortfall_bps:>10.2f} bps")
    print(f"  VWAP Slippage (bps)        : {result.vwap_slippage_bps:>10.2f} bps")
    print(f"  Market Impact Cost         : ${result.market_impact_cost:>14,.2f}")
    print(f"  Transaction Fees           : ${result.total_transaction_fees:>14,.2f}")
    print(f"  Terminal Penalty           : ${result.terminal_penalty:>14,.2f}")
    print(f"  Total Execution Cost       : ${result.execution_cost:>14,.2f}")


def print_comparison_table(results: list[BaselineResult]) -> None:
    metrics = [
        ("Completion Rate (%)",   lambda r: f"{r.completion_rate*100:.1f}"),
        ("Avg Exec Price ($)",    lambda r: f"{r.average_execution_price:.4f}"),
        ("IS (bps)",              lambda r: f"{r.implementation_shortfall_bps:.2f}"),
        ("VWAP Slippage (bps)",   lambda r: f"{r.vwap_slippage_bps:.2f}"),
        ("Market Impact ($)",     lambda r: f"{r.market_impact_cost:,.2f}"),
        ("Fees ($)",              lambda r: f"{r.total_transaction_fees:,.2f}"),
        ("Terminal Penalty ($)",  lambda r: f"{r.terminal_penalty:,.2f}"),
        ("Total Cost ($)",        lambda r: f"{r.execution_cost:,.2f}"),
        ("Turnover ($)",          lambda r: f"{r.turnover:,.0f}"),
        ("Active Steps",          lambda r: f"{r.execution_duration_steps}"),
    ]

    names = [r.strategy_name for r in results]
    col_w = max(len(n) for n in names) + 2
    label_w = 22

    header = f"{'Metric':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    sep    = "-" * len(header)

    print(f"\n\n{'='*60}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(sep)
    print(header)
    print(sep)
    for label, fn in metrics:
        row = f"{label:<{label_w}}" + "".join(f"{fn(r):>{col_w}}" for r in results)
        print(row)
    print(sep)


def main():
    print("=" * 60)
    print("  EXECUTION BASELINE COMPARISON")
    print(f"  Asset: AAPL | Side: {SIDE} | Target: {TARGET_INVENTORY:,.0f} shares")
    print(f"  Horizon: {HORIZON_STEPS} min | 1-min bars")
    print("=" * 60)

    df = load_data()

    # Shared impact model — identical for all strategies
    impact = AlmgrenChrissImpactModel(eta=0.05, gamma=0.01, alpha=0.5)
    runner = BaselineRunner(
        impact_model=impact,
        max_participation_rate=0.15,
        per_share_fee=0.0005,
    )

    strategies = [
        TWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        VWAPStrategy(target_inventory=TARGET_INVENTORY, total_steps=HORIZON_STEPS),
        POVStrategy(target_inventory=TARGET_INVENTORY, target_rate=0.10),
        POVStrategy(target_inventory=TARGET_INVENTORY, target_rate=0.20),
    ]

    results = []
    for strat in strategies:
        r = runner.run(
            strategy=strat,
            market_data=df,
            target_inventory=TARGET_INVENTORY,
            side=SIDE,
            horizon_steps=HORIZON_STEPS,
        )
        results.append(r)
        print_result(r)

    print_comparison_table(results)
    print()


if __name__ == "__main__":
    main()
