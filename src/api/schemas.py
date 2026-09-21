"""
src/api/schemas.py — Pydantic Schemas for KAIRO Execution Intelligence API

Defines all request and response Data Transfer Objects (DTOs) for:
  - Single execution simulation (/api/execution/simulate)
  - Baseline & DRL backtest comparison (/api/execution/backtest)
  - Execution run querying by ID (/api/execution/{id})
  - Metrics retrieval (/api/execution/{id}/metrics)
  - Trajectory retrieval (/api/execution/{id}/trajectory)
  - Current market regime detection (/api/regime/current)
  - Metadata queries (/api/baselines, /api/models, /api/experiments)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Schemas ─────────────────────────────────────────────────────────────

class ExecutionSimulateRequest(BaseModel):
    """Payload for POST /api/execution/simulate."""
    symbol: str = Field(default="AAPL", description="Ticker symbol")
    side: str = Field(default="BUY", description="Order side: 'BUY' or 'SELL'")
    quantity: float = Field(default=100_000.0, gt=0, le=10_000_000, description="Target execution inventory quantity")
    horizon_steps: int = Field(default=30, ge=5, le=120, description="Total execution horizon in one-minute steps")
    policy: str = Field(
        default="DQN",
        description="Policy to execute: TWAP, VWAP, POV, DQN, Regime-Aware DQN, PPO, Regime-Aware PPO",
    )
    scenario: str = Field(
        default="normal",
        description="Market scenario: normal, high_volatility, low_liquidity, stress, regime_transition, liquidity_shock",
    )
    seed: int = Field(default=42, description="Random seed for deterministic execution rollout")


class BacktestRequest(BaseModel):
    """Payload for POST /api/execution/backtest."""
    symbol: str = Field(default="AAPL", description="Ticker symbol")
    side: str = Field(default="BUY", description="Order side: 'BUY' or 'SELL'")
    quantity: float = Field(default=100_000.0, gt=0, le=10_000_000, description="Target execution inventory quantity")
    horizon_steps: int = Field(default=30, ge=5, le=120, description="Total execution horizon in one-minute steps")
    n_windows: int = Field(default=10, ge=1, le=30, description="Number of out-of-sample windows to average over")
    policies: List[str] = Field(
        default=["TWAP", "VWAP", "POV", "DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"],
        description="List of policies/baselines to run in the backtest comparison",
    )
    scenario: str = Field(
        default="normal",
        description="Market scenario for backtest comparison",
    )
    seed: int = Field(default=42, description="Random seed for backtest repeatability")


# ── Metric & Trajectory Schemas ─────────────────────────────────────────────────

class ExecutionMetricsResponse(BaseModel):
    """Standardised execution quality metrics (mirrors BaselineResult / AgentEvalResult)."""
    implementation_shortfall: float = Field(description="Implementation Shortfall in currency ($)")
    implementation_shortfall_bps: float = Field(description="Implementation Shortfall in basis points (bps)")
    execution_cost: float = Field(description="Total execution cost ($)")
    market_impact_cost: float = Field(description="Market impact cost component ($)")
    total_transaction_fees: float = Field(description="Transaction fees incurred ($)")
    terminal_penalty: float = Field(description="Terminal unexecuted inventory penalty ($)")
    completion_rate: float = Field(description="Fraction of target inventory executed [0, 1]")
    average_execution_price: float = Field(description="Weighted average fill price")
    arrival_price: float = Field(description="Decision price at execution start")
    market_vwap_price: float = Field(description="Benchmark market VWAP price over the horizon")
    vwap_slippage_bps: float = Field(description="Execution price slippage vs. market VWAP (bps)")
    action_counts: Dict[str, int] = Field(default_factory=dict, description="Action index execution counts")


class ExecutionTrajectoryResponse(BaseModel):
    """Step-by-step trajectory array for execution monitoring and visualization."""
    execution_id: str
    inventory_trajectory: List[float] = Field(description="Remaining inventory at each step")
    action_trajectory: List[int] = Field(description="Action index selected at each step")
    price_trajectory: List[float] = Field(description="Market price trajectory at each step")
    regime_trajectory: Optional[List[int]] = Field(default=None, description="HMM market regime ID at each step")
    observation_trajectory: Optional[List[List[float]]] = Field(
        default=None, description="Observation vector the policy saw at each step (learned policies only)")
    window_start: Optional[int] = Field(default=None, description="Bar index at which the execution window starts")


# ── Execution Response Schemas ──────────────────────────────────────────────────

class ExecutionResponse(BaseModel):
    """Detailed response object returned for simulation run or GET /api/execution/{id}."""
    execution_id: str
    timestamp: str
    symbol: str
    side: str
    target_inventory: float
    executed_inventory: float
    remaining_inventory: float
    policy: str
    scenario: str
    status: str = Field(default="completed", description="'completed' when fully filled, 'partial' otherwise")
    metrics: ExecutionMetricsResponse


class BacktestResultItem(BaseModel):
    """Summary item for a single strategy inside a backtest response (mean over windows)."""
    policy: str
    implementation_shortfall_bps: float
    implementation_shortfall_bps_std: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    vs_twap_bps: Optional[float] = Field(default=None, description="Paired mean IS difference vs TWAP (negative = cheaper)")
    vs_twap_ci_low: Optional[float] = None
    vs_twap_ci_high: Optional[float] = None
    execution_cost: float
    completion_rate: float
    vwap_slippage_bps: float
    n_windows: int = 1


class BacktestError(BaseModel):
    """A policy that could not be evaluated, with the reason."""
    policy: str
    detail: str


class BacktestResponse(BaseModel):
    """Response object returned for POST /api/execution/backtest."""
    backtest_id: str
    timestamp: str
    symbol: str
    side: str
    quantity: float
    scenario: str
    n_windows: int = 1
    results: List[BacktestResultItem]
    errors: List[BacktestError] = Field(default_factory=list)


# ── Regime & Metadata Schemas ───────────────────────────────────────────────────

class CurrentRegimeResponse(BaseModel):
    """Response payload for GET /api/regime/current."""
    symbol: str
    timestamp: str
    current_price: float
    spread: float
    volatility: float
    regime_id: int
    regime_label: str
    regime_probabilities: Dict[str, float]
    true_regime: Optional[str] = Field(default=None, description="Latent regime of the synthetic market (validation only)")


class BaselineStrategyResponse(BaseModel):
    """Item for GET /api/baselines."""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]


class ModelMetadataResponse(BaseModel):
    """Item for GET /api/models."""
    model_id: str
    name: str
    algorithm_class: str
    regime_aware: bool
    status: str = Field(description="'trained' when a checkpoint exists, otherwise 'untrained'")
    trained: bool = False
    state_dim: int
    net_arch: List[int]
    timesteps: Optional[int] = None
    trained_at: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Held-out evaluation recorded at training time")


class ExperimentResultsResponse(BaseModel):
    """Response for GET /api/experiments/results: the recorded research results."""
    config: Dict[str, Any]
    summary: List[Dict[str, Any]] = Field(description="Per-strategy IS statistics across seeds")
    comparisons: List[Dict[str, Any]] = Field(description="Paired comparisons with confidence intervals")
    hmm_validation: Dict[str, Optional[float]] = Field(default_factory=dict)


class ExperimentSummaryResponse(BaseModel):
    """Item for GET /api/experiments."""
    experiment_id: str
    name: str
    scenarios: List[str]
    strategies: List[str]
    status: str = Field(description="'completed' only when result files exist on disk, otherwise 'not_run'")
    results_available: bool = False
    seeds: Optional[List[int]] = None
    train_timesteps: Optional[int] = None


# ── Decision Explanation Schemas ────────────────────────────────────────────────

class DecisionExplanationRequest(BaseModel):
    """Payload for POST /api/execution/explain."""
    state: List[float] = Field(description="Observation state vector (7-dim standard or 12-dim regime-aware)")
    action: Optional[int] = Field(default=None, ge=0, le=3,
                                  description="Action to explain; defaults to the policy's own greedy action")
    policy: str = Field(default="Regime-Aware DQN", description="Learned policy whose trained network is explained")


class DecisionExplanationResponse(BaseModel):
    """Response payload for decision explanation endpoint."""
    action: int
    action_label: str
    feature_attributions: Dict[str, float]
    feature_percentages: Dict[str, float]
    regime_influence_score: float
    action_scores: Dict[int, float]
    action_advantages: Dict[int, float]
    summary: str


# ── Paper Trading & Risk Gate Schemas ───────────────────────────────────────────

class PaperOrderRequest(BaseModel):
    """Payload for POST /api/execution/paper."""
    symbol: str = Field(default="AAPL", description="Ticker symbol")
    side: str = Field(default="BUY", description="Order side ('BUY' or 'SELL')")
    quantity: float = Field(default=10_000.0, gt=0, description="Slice shares to execute")
    current_price: float = Field(default=150.0, gt=0, description="Current market price ($)")
    arrival_price: float = Field(default=150.0, gt=0, description="Decision arrival price ($)")
    target_inventory: float = Field(default=100_000.0, gt=0, description="Total target order inventory")


class PaperOrderResponse(BaseModel):
    """Response payload for paper order submission."""
    order_id: Optional[str]
    status: str
    mode: str = "paper_mock"
    symbol: str
    side: str
    executed_quantity: float
    fill_price: float
    reason: str
    timestamp: str


class RiskStatusResponse(BaseModel):
    """Response payload for GET /api/execution/risk-status."""
    max_notional_value: float
    max_single_order_pct: float
    price_collar_pct: float
    kill_switch_active: bool


class SetKillSwitchRequest(BaseModel):
    """Payload for POST /api/execution/kill-switch."""
    active: bool = Field(description="Set to true to trigger emergency halt kill switch")


