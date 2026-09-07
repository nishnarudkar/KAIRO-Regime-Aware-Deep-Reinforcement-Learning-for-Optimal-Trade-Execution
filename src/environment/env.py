"""
Gymnasium-compatible MDP Environment for Optimal Trade Execution.

Class: TradeExecutionEnv
Wraps ExecutionSimulator with standard Gymnasium interfaces (reset, step, action_space, observation_space).
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.execution import (
    ExecutionSimulator,
    BaseImpactModel,
    AlmgrenChrissImpactModel,
    StepResult
)
from src.environment.rewards import ModularExecutionReward


class TradeExecutionEnv(gym.Env):
    """
    Gymnasium Trade Execution MDP Environment.
    
    Observation Space: Box(7,)
      0: log_return
      1: volatility
      2: volume_ratio (volume / rolling_mean_volume)
      3: spread_ratio (spread / price)
      4: liquidity_proxy (volume / (spread + eps))
      5: remaining_inventory_fraction (q_t / Q_0)
      6: time_remaining_fraction ((T - t) / T)

    Action Space: Discrete(4)
      0: execute 0.00 (0% of remaining inventory)
      1: execute 0.10 (10% of remaining inventory)
      2: execute 0.25 (25% of remaining inventory)
      3: execute 0.50 (50% of remaining inventory)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: Optional[pd.DataFrame] = None,
        target_inventory: float = 100000.0,
        side: str = "BUY",
        horizon_steps: int = 30,
        action_fractions: List[float] = [0.0, 0.10, 0.25, 0.50],
        impact_model: Optional[BaseImpactModel] = None,
        reward_calculator: Optional[ModularExecutionReward] = None,
        max_participation_rate: float = 0.15,
        per_share_fee: float = 0.0005,
        default_spread_bps: float = 2.0,
    ):
        super().__init__()

        self.target_inventory = float(target_inventory)
        self.side = side.upper()
        self.horizon_steps = int(horizon_steps)
        self.action_fractions = action_fractions
        self.num_actions = len(action_fractions)

        # Gymnasium Action and Observation Spaces
        self.action_space = spaces.Discrete(self.num_actions)

        # 7-dimensional continuous observation vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # Instantiate Impact Model and Simulator
        self.impact_model = impact_model if impact_model is not None else AlmgrenChrissImpactModel(eta=0.05, gamma=0.01)
        self.simulator = ExecutionSimulator(
            impact_model=self.impact_model,
            max_participation_rate=max_participation_rate,
            per_share_fee=per_share_fee,
            default_spread_bps=default_spread_bps
        )

        # Reward calculator
        self.reward_calculator = reward_calculator if reward_calculator is not None else ModularExecutionReward()

        # Market data store
        self.market_data = market_data
        self._prev_price: float = 0.0
        self._mean_volume: float = 1.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset Gymnasium environment to start a new execution episode.
        
        Args:
            seed: Random seed for deterministic reproducibility.
            options: Dictionary containing optional override 'market_data' or 'target_inventory'.
            
        Returns:
            Tuple of (initial_observation_vector, info_dict).
        """
        super().reset(seed=seed)

        # Extract market data from options if passed, or default
        df = None
        if options and "market_data" in options:
            df = options["market_data"]
        elif self.market_data is not None:
            df = self.market_data
        else:
            # Fallback synthetic deterministic market data if none provided
            df = self._generate_fallback_data()

        target_inv = options.get("target_inventory", self.target_inventory) if options else self.target_inventory

        # Reset simulator
        sim_state = self.simulator.reset(
            market_data=df,
            target_inventory=target_inv,
            side=self.side,
            horizon_steps=self.horizon_steps
        )

        self._prev_price = sim_state["current_price"]
        self._mean_volume = float(df['volume'].mean()) if 'volume' in df.columns and len(df) > 0 else 10000.0

        obs = self._get_observation(sim_state)
        info = self._get_info(sim_state)

        return obs, info

    def step(self, action: Union[int, np.integer]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform single Gymnasium environment step.
        
        Args:
            action: Discrete action index in [0, 1, 2, 3].
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be an integer in range [0, {self.num_actions-1}].")

        fraction = self.action_fractions[int(action)]
        rem_inventory = self.simulator.remaining_inventory
        order_qty = fraction * rem_inventory

        # Execute step in simulator
        step_result = self.simulator.step(order_qty=order_qty)
        sim_state = self.simulator.get_state()

        # Termination checks
        terminated = sim_state["is_done"]
        truncated = False  # Truncation can occur if market replay buffer ends early

        # Calculate reward
        terminal_penalty = 0.0
        if terminated and self.simulator.remaining_inventory > 0:
            metrics = self.simulator.compute_metrics()
            terminal_penalty = metrics.terminal_penalty

        reward_dict = self.reward_calculator.calculate_step_reward(
            step_result=step_result,
            arrival_price=self.simulator.arrival_price,
            target_inventory=self.simulator.target_inventory,
            volatility=sim_state["volatility"],
            is_terminal=terminated,
            terminal_penalty=terminal_penalty
        )
        reward = reward_dict["reward"]

        obs = self._get_observation(sim_state)
        info = self._get_info(sim_state, step_result=step_result, reward_dict=reward_dict)

        self._prev_price = sim_state["current_price"]

        return obs, reward, terminated, truncated, info

    def _get_observation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """Construct 7-dimensional continuous state vector."""
        current_price = sim_state["current_price"]
        prev_price = self._prev_price if self._prev_price > 0 else current_price

        # 0. Log return
        log_return = np.log(current_price / prev_price) if prev_price > 0 else 0.0

        # 1. Volatility
        volatility = sim_state["volatility"]

        # 2. Relative Volume
        volume = sim_state["volume"]
        volume_ratio = volume / self._mean_volume if self._mean_volume > 0 else 1.0

        # 3. Relative Spread
        spread = sim_state["spread"]
        spread_ratio = spread / current_price if current_price > 0 else 0.0002

        # 4. Liquidity Proxy
        liquidity_proxy = volume / (spread + 1e-6)

        # 5. Remaining Inventory Fraction
        remaining_frac = sim_state["remaining_inventory"] / sim_state["target_inventory"]

        # 6. Time Remaining Fraction
        time_remaining_frac = sim_state["remaining_time"] / max(1, self.horizon_steps)

        obs = np.array([
            log_return,
            volatility,
            volume_ratio,
            spread_ratio,
            liquidity_proxy,
            remaining_frac,
            time_remaining_frac
        ], dtype=np.float32)

        return obs

    def _get_info(
        self,
        sim_state: Dict[str, Any],
        step_result: Optional[StepResult] = None,
        reward_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Construct info dictionary exposing research metrics without lookahead leakage."""
        info = {
            "current_step": sim_state["current_step"],
            "timestamp": str(sim_state["current_timestamp"]),
            "current_price": sim_state["current_price"],
            "remaining_inventory": sim_state["remaining_inventory"],
            "executed_inventory": sim_state["executed_inventory"],
            "remaining_time": sim_state["remaining_time"],
            "average_execution_price": sim_state["average_execution_price"],
        }

        if step_result is not None:
            info.update({
                "filled_qty": step_result.filled_qty,
                "execution_price": step_result.execution_price,
                "temporary_impact": step_result.temporary_impact,
                "permanent_impact": step_result.permanent_impact,
                "step_cost": step_result.step_cost,
                "transaction_fee": step_result.transaction_fee,
            })

        if reward_dict is not None:
            info.update({
                "reward_cost_penalty": reward_dict["cost_penalty"],
                "reward_impact_penalty": reward_dict["impact_penalty"],
                "reward_risk_penalty": reward_dict["risk_penalty"],
                "reward_fee_penalty": reward_dict["fee_penalty"],
                "reward_terminal_penalty": reward_dict["terminal_penalty"],
            })

        if sim_state["is_done"]:
            summary = self.simulator.compute_metrics()
            info.update({
                "is_bps": summary.implementation_shortfall_bps,
                "vwap_slippage_bps": summary.vwap_slippage_bps,
                "fill_rate": summary.fill_rate,
                "total_execution_cost": summary.total_execution_cost,
                "terminal_penalty": summary.terminal_penalty,
            })

        return info

    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate deterministic fallback market data if none provided."""
        dates = pd.date_range("2026-01-01 09:30", periods=self.horizon_steps, freq="1min")
        prices = 150.0 + np.sin(np.linspace(0, 3, self.horizon_steps)) * 0.5
        volumes = [50000] * self.horizon_steps
        spreads = [0.04] * self.horizon_steps
        vols = [0.0015] * self.horizon_steps

        df = pd.DataFrame({
            "timestamp": dates,
            "price": prices,
            "volume": volumes,
            "spread": spreads,
            "volatility": vols
        })
        return df
