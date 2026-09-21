"""
Gymnasium-compatible MDP Environment for Optimal Trade Execution.

Class: TradeExecutionEnv
Wraps ExecutionSimulator with standard Gymnasium interfaces (reset, step, action_space, observation_space).
"""

from typing import Dict, Any, Optional, Tuple, List, Sequence, Union
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.execution import (
    ExecutionSimulator,
    BaseImpactModel,
    default_impact_model,
    StepResult
)
from src.environment.rewards import ModularExecutionReward, default_reward_calculator


class TradeExecutionEnv(gym.Env):
    """
    Gymnasium Trade Execution MDP Environment.
    
    Observation Space: Box(7,)  (fixed unit conversions, no data-dependent scaling)
      0: log_return            one-bar log return, in percent
      1: volatility            causal realized volatility, in percent
      2: volume_ratio          bar volume / mean volume of the preceding history
      3: spread_ratio          quoted spread / price, in basis points
      4: liquidity_proxy       ln(1 + volume / spread) - 13   (roughly centred)
      5: remaining_inventory_fraction (q_t / Q_0)
      6: time_remaining_fraction ((T - t) / T)

    Episode windows:
      By default an episode replays bars [0, horizon) of the data. With
      ``random_start=True`` each episode starts at a random bar inside
      ``start_range`` (in a random dataset when ``market_data`` is a list), so
      agents see many different market paths. Bars before the start are
      available as *history* only, never the future.

    Action Space: Discrete(4)
      0: execute 0.00 (0% of remaining inventory)
      1: execute 0.10 (10% of remaining inventory)
      2: execute 0.25 (25% of remaining inventory)
      3: execute 0.50 (50% of remaining inventory)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: Optional[Union[pd.DataFrame, Sequence[pd.DataFrame]]] = None,
        target_inventory: float = 100000.0,
        side: str = "BUY",
        horizon_steps: int = 30,
        action_fractions: List[float] = [0.0, 0.10, 0.25, 0.50],
        impact_model: Optional[BaseImpactModel] = None,
        reward_calculator: Optional[ModularExecutionReward] = None,
        max_participation_rate: float = 0.15,
        per_share_fee: float = 0.0005,
        default_spread_bps: float = 2.0,
        random_start: bool = False,
        start_range: Optional[Tuple[int, int]] = None,
        history_bars: int = 60,
        order_participation: Optional[float] = None,
    ):
        super().__init__()

        # If set, each episode's order is sized as this fraction of the *expected* volume of the
        # window (trailing mean volume x horizon), so orders are comparable across liquidity levels
        # and fillable under the participation cap. Otherwise ``target_inventory`` is used.
        self.order_participation = order_participation
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
        self.impact_model = impact_model if impact_model is not None else default_impact_model()
        self.simulator = ExecutionSimulator(
            impact_model=self.impact_model,
            max_participation_rate=max_participation_rate,
            per_share_fee=per_share_fee,
            default_spread_bps=default_spread_bps
        )

        # Reward calculator
        self.reward_calculator = reward_calculator if reward_calculator is not None else default_reward_calculator()

        # Market data store (a single DataFrame or a list of them)
        self.market_data = market_data
        if isinstance(market_data, (list, tuple)):
            self._datasets: List[pd.DataFrame] = list(market_data)
        elif market_data is not None:
            self._datasets = [market_data]
        else:
            self._datasets = []
        self.random_start = random_start
        self.start_range = start_range
        self.history_bars = int(history_bars)

        self._prev_price: float = 0.0
        self._mean_volume: float = 1.0
        self.window_start: int = 0
        self.dataset_idx: int = 0
        self.episode_target: float = float(target_inventory)

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

        df, start = self._resolve_episode(options)
        self.window_start = start
        window = df.iloc[start:start + self.horizon_steps]

        # Causal reference quantities: only bars *before* the window start are used.
        history = df.iloc[max(0, start - self.history_bars):start]
        if len(history) >= 5 and "volume" in history.columns:
            self._mean_volume = float(history["volume"].mean())
        else:
            self._mean_volume = float(window["volume"].iloc[0]) if len(window) > 0 else 10000.0

        if options and "target_inventory" in options:
            target_inv = float(options["target_inventory"])
        elif self.order_participation:
            target_inv = float(self.order_participation * self._mean_volume * self.horizon_steps)
        else:
            target_inv = self.target_inventory
        self.episode_target = target_inv

        # Reset simulator on the episode window
        sim_state = self.simulator.reset(
            market_data=window,
            target_inventory=target_inv,
            side=self.side,
            horizon_steps=self.horizon_steps
        )
        if len(history) > 0:
            price_col = "price" if "price" in history.columns else "close"
            self._prev_price = float(history[price_col].iloc[-1])
        else:
            self._prev_price = sim_state["current_price"]

        obs = self._get_observation(sim_state)
        info = self._get_info(sim_state)

        return obs, info

    def _resolve_episode(self, options: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, int]:
        """Pick the dataset and the window start for the next episode."""
        options = options or {}
        if "market_data" in options:
            datasets = [options["market_data"]]
        elif self._datasets:
            datasets = self._datasets
        else:
            datasets = [self._generate_fallback_data()]

        if "dataset_idx" in options:
            ds_idx = int(options["dataset_idx"])
        elif len(datasets) > 1:
            ds_idx = int(self.np_random.integers(len(datasets)))
        else:
            ds_idx = 0
        self.dataset_idx = ds_idx
        df = datasets[ds_idx]

        max_start = max(0, len(df) - self.horizon_steps)
        if "start_idx" in options:
            start = int(options["start_idx"])
        elif self.random_start:
            lo, hi = self.start_range if self.start_range else (0, max_start)
            lo, hi = max(0, lo), min(max_start, hi)
            start = int(self.np_random.integers(lo, hi + 1)) if hi >= lo else 0
        elif self.start_range:
            start = self.start_range[0]
        else:
            start = 0
        return df, min(max(start, 0), max_start)

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

        # 0. Log return (percent)
        log_return = 100.0 * np.log(current_price / prev_price) if prev_price > 0 else 0.0

        # 1. Volatility (percent)
        volatility = 100.0 * sim_state["volatility"]

        # 2. Relative Volume
        volume = sim_state["volume"]
        volume_ratio = volume / self._mean_volume if self._mean_volume > 0 else 1.0

        # 3. Relative Spread (basis points)
        spread = sim_state["spread"]
        spread_ratio = 1e4 * spread / current_price if current_price > 0 else 2.0

        # 4. Liquidity Proxy (log scale, roughly centred)
        liquidity_proxy = np.log1p(volume / (spread + 1e-6)) - 13.0

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
