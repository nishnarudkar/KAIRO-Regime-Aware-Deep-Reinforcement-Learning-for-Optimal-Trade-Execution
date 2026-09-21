"""
Modular Reward Calculator for Trade Execution MDP.

Calculates step-by-step rewards incorporating:
- Execution shortfall cost
- Temporary market impact cost
- Inventory holding risk
- Transaction fees
- Terminal incomplete order penalty
"""

from typing import Dict, Any, Optional
import numpy as np

from src.environment import BaseRewardCalculator
from src.execution.simulator import StepResult


class ModularExecutionReward(BaseRewardCalculator):
    """
    Modular reward calculator combining weighted execution penalties.
    
    Formula:
      R_t = - ( lambda_cost * Cost_t + lambda_impact * Impact_t +
                lambda_risk * Risk_t + lambda_fee * Fee_t +
                lambda_terminal * TerminalPenalty_t )
    """

    def __init__(
        self,
        lambda_cost: float = 1.0,
        lambda_impact: float = 1.0,
        lambda_risk: float = 1e-4,
        lambda_fee: float = 1.0,
        lambda_terminal: float = 10.0,
        reward_scale: float = 1.0,
    ):
        """
        Args:
            lambda_cost: Weight for execution shortfall cost penalty.
            lambda_impact: Weight for temporary market impact cost penalty.
            lambda_risk: Weight for inventory holding risk penalty.
            lambda_fee: Weight for transaction fee penalty.
            lambda_terminal: Weight for terminal unexecuted inventory penalty.
            reward_scale: Multiplier applied to the final reward. Penalties are fractions of
                          order notional (~1e-4 per step), which starves gradient-based agents;
                          scaling by 1e4 expresses the reward in basis points of notional.
        """
        self.lambda_cost = lambda_cost
        self.lambda_impact = lambda_impact
        self.lambda_risk = lambda_risk
        self.lambda_fee = lambda_fee
        self.lambda_terminal = lambda_terminal
        self.reward_scale = reward_scale

    def calculate_reward(
        self,
        execution_price: float,
        benchmark_price: float,
        executed_qty: float,
        remaining_qty: float,
        step: int,
        max_steps: int
    ) -> float:
        """Standard interface fallback method."""
        # Simple shortfall penalty normalized by benchmark price
        if benchmark_price <= 0:
            return 0.0
        cost = executed_qty * (execution_price - benchmark_price) / benchmark_price
        risk = (remaining_qty ** 2) * 1e-6
        return -float(self.lambda_cost * cost + self.lambda_risk * risk)

    def calculate_step_reward(
        self,
        step_result: StepResult,
        arrival_price: float,
        target_inventory: float,
        volatility: float,
        is_terminal: bool = False,
        terminal_penalty: float = 0.0,
        benchmark_price: float = None,
    ) -> Dict[str, float]:
        """
        Calculate detailed step reward components.
        
        Args:
            step_result: StepResult object from ExecutionSimulator.
            arrival_price: Initial arrival mid price benchmark P_0.
            target_inventory: Total parent order target Q_0.
            volatility: Current bar volatility sigma_t.
            is_terminal: Whether episode has reached terminal state.
            terminal_penalty: Absolute currency terminal penalty for unfilled inventory.
            
        Returns:
            Dictionary containing total reward and individual component values.
        """
        if arrival_price <= 0 or target_inventory <= 0:
            return {
                "reward": 0.0,
                "cost_penalty": 0.0,
                "impact_penalty": 0.0,
                "risk_penalty": 0.0,
                "fee_penalty": 0.0,
                "terminal_penalty": 0.0
            }

        norm_base = target_inventory * arrival_price

        # 1. Execution Shortfall Cost Penalty (normalized by Q_0 * P_0).
        #    By default the fill is charged against the arrival price (true implementation shortfall).
        #    With ``benchmark_price`` (the un-impacted market price at this step) the uncontrollable price
        #    drift since arrival is removed: what remains is spread + temporary impact + the permanent impact
        #    of earlier trades, a far less noisy learning signal (a control variate; evaluation still uses the
        #    true arrival-price shortfall).
        reference = arrival_price if benchmark_price is None else benchmark_price
        shortfall_cost = step_result.filled_qty * (step_result.execution_price - reference)
        cost_penalty = self.lambda_cost * (shortfall_cost / norm_base)

        # 2. Temporary Market Impact Penalty
        impact_cost = step_result.filled_qty * abs(step_result.temporary_impact)
        impact_penalty = self.lambda_impact * (impact_cost / norm_base)

        # 3. Inventory Risk Penalty: (q_t / Q_0)^2 * volatility^2
        remaining_frac = step_result.remaining_inventory / target_inventory
        risk_cost = (remaining_frac ** 2) * (volatility ** 2)
        risk_penalty = self.lambda_risk * risk_cost

        # 4. Fee Penalty
        fee_penalty = self.lambda_fee * (step_result.transaction_fee / norm_base)

        # 5. Terminal Incomplete Penalty
        if is_terminal and terminal_penalty > 0:
            term_penalty = self.lambda_terminal * (terminal_penalty / norm_base)
        else:
            term_penalty = 0.0

        total_penalty = cost_penalty + impact_penalty + risk_penalty + fee_penalty + term_penalty
        reward = -float(total_penalty) * self.reward_scale

        return {
            "reward": reward,
            "cost_penalty": cost_penalty,
            "impact_penalty": impact_penalty,
            "risk_penalty": risk_penalty,
            "fee_penalty": fee_penalty,
            "terminal_penalty": term_penalty
        }


DEFAULT_REWARD_SCALE = 1.0e4   # reward expressed in bps of parent-order notional


def default_reward_calculator() -> ModularExecutionReward:
    """Reward used by every environment unless one is passed explicitly."""
    return ModularExecutionReward(reward_scale=DEFAULT_REWARD_SCALE)
