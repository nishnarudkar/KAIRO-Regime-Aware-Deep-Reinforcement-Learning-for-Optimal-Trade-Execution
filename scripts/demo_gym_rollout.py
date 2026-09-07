"""
Random-Action Rollout Demo for TradeExecutionEnv Gymnasium MDP.

Scenario:
- Target Order: 100,000 shares AAPL
- Horizon: 30 minutes
- Policy: Uniform random action sampling from Discrete(4)
"""

import os
import pandas as pd
import numpy as np

from src.environment import TradeExecutionEnv, ModularExecutionReward

def run_gym_rollout_demo():
    print("=" * 75)
    print("      GYMNASIUM TRADE EXECUTION MDP - RANDOM ROLLOUT DEMO")
    print("=" * 75)

    # 1. Load market replay dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_market_data.csv")
    market_df = pd.read_csv(data_path)
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])

    # 2. Instantiate Gymnasium Environment
    env = TradeExecutionEnv(
        market_data=market_df,
        target_inventory=100000.0,
        side="BUY",
        horizon_steps=30
    )

    # 3. Reset Environment with deterministic seed
    obs, info = env.reset(seed=42)

    print(f"\n[ENVIRONMENT INITIALIZED]")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Action Space:      {env.action_space} (0: 0%, 1: 10%, 2: 25%, 3: 50%)")
    print(f"  Initial State Vector (shape {obs.shape}):\n    {obs}")

    print("\n" + "-" * 75)
    print(f"{'Step':<5} | {'Action':<6} | {'Fraction':<8} | {'Filled':<8} | {'Remaining':<10} | {'Reward':<9} | {'Exec Price':<10}")
    print("-" * 75)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    action_fraction_map = {0: "0%", 1: "10%", 2: "25%", 3: "50%"}

    while not (terminated or truncated):
        # Sample random action
        action = int(env.action_space.sample())
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        total_reward += reward

        print(
            f"{info['current_step']-1:<5} | "
            f"{action:<6} | "
            f"{action_fraction_map[action]:<8} | "
            f"{info.get('filled_qty', 0):>8.0f} | "
            f"{info['remaining_inventory']:>10.0f} | "
            f"{reward:>9.4f} | "
            f"${info.get('execution_price', 0):>9.2f}"
        )

    print("=" * 75)
    print("      ROLLOUT COMPLETED - FINAL MDP SUMMARY")
    print("=" * 75)
    print(f"  Total Steps Taken:              {step_count}")
    print(f"  Cumulative Episode Reward:      {total_reward:.4f}")
    print(f"  Final Executed Inventory:       {info.get('executed_inventory', 0):,.0f} shares")
    print(f"  Final Remaining Inventory:      {info.get('remaining_inventory', 0):,.0f} shares")
    print(f"  Fill Rate:                      {info.get('fill_rate', 0)*100:.1f}%")
    print(f"  Implementation Shortfall (bps):  {info.get('is_bps', 0):.2f} bps")
    print(f"  VWAP Slippage (bps):            {info.get('vwap_slippage_bps', 0):.2f} bps")
    print(f"  Terminal Penalty ($):           ${info.get('terminal_penalty', 0):,.2f}")
    print("=" * 75)

if __name__ == "__main__":
    run_gym_rollout_demo()
