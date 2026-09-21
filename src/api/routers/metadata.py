"""
src/api/routers/metadata.py — Platform Metadata Routes

Endpoints:
  GET /api/baselines   — List available baseline strategies (TWAP, VWAP, POV)
  GET /api/models      — List available DRL policies (DQN, Regime DQN, PPO, Regime PPO)
  GET /api/experiments — List available research experiment suites & scenarios
"""

from __future__ import annotations

from typing import List
from fastapi import APIRouter

from src.api.schemas import (
    BaselineStrategyResponse,
    ModelMetadataResponse,
    ExperimentSummaryResponse,
)
from src.evaluation.scenarios import SCENARIOS

router = APIRouter(tags=["Platform Metadata"])


@router.get("/api/baselines", response_model=List[BaselineStrategyResponse])
def list_baselines():
    """List all available conventional execution baseline strategies."""
    return [
        BaselineStrategyResponse(
            strategy_id="TWAP",
            name="Time-Weighted Average Price",
            description="Executes inventory uniformly across horizon steps: v_t = Target / T",
            parameters={"target_inventory": 100000.0, "total_steps": 30},
        ),
        BaselineStrategyResponse(
            strategy_id="VWAP",
            name="Volume-Weighted Average Price",
            description="Executes inventory proportional to historical volume profile: v_t = Target * (V_t / sum(V))",
            parameters={"target_inventory": 100000.0, "total_steps": 30},
        ),
        BaselineStrategyResponse(
            strategy_id="POV",
            name="Percentage of Volume",
            description="Executes fixed percentage (e.g. 10%) of prevailing market volume at each step: v_t = 0.10 * V_t",
            parameters={"target_inventory": 100000.0, "target_rate": 0.10},
        ),
    ]


@router.get("/api/models", response_model=List[ModelMetadataResponse])
def list_models():
    """List all available trained Deep Reinforcement Learning execution models."""
    return [
        ModelMetadataResponse(
            model_id="dqn_base",
            name="DQN (no regime)",
            algorithm_class="Value-Based / Off-Policy",
            regime_aware=False,
            status="validated",
            state_dim=7,
            net_arch=[128, 128],
        ),
        ModelMetadataResponse(
            model_id="dqn_regime",
            name="Regime-Aware DQN",
            algorithm_class="Value-Based / Off-Policy",
            regime_aware=True,
            status="validated",
            state_dim=12,
            net_arch=[128, 128],
        ),
        ModelMetadataResponse(
            model_id="ppo_base",
            name="PPO (no regime)",
            algorithm_class="Policy-Gradient / On-Policy",
            regime_aware=False,
            status="validated",
            state_dim=7,
            net_arch=[128, 128],
        ),
        ModelMetadataResponse(
            model_id="ppo_regime",
            name="Regime-Aware PPO",
            algorithm_class="Policy-Gradient / On-Policy",
            regime_aware=True,
            status="validated",
            state_dim=12,
            net_arch=[128, 128],
        ),
    ]


@router.get("/api/experiments", response_model=List[ExperimentSummaryResponse])
def list_experiments():
    """List available research experiment suites and market scenarios."""
    scenarios = list(SCENARIOS.keys())
    return [
        ExperimentSummaryResponse(
            experiment_id="stage8_research_suite",
            name="Stage 8: Research Experiment Suite",
            scenarios=scenarios,
            strategies=["TWAP", "VWAP", "POV", "DQN", "Regime-Aware DQN"],
            status="completed",
        ),
        ExperimentSummaryResponse(
            experiment_id="stage9_ppo_extension",
            name="Stage 9: Multi-Algorithm PPO Extension",
            scenarios=scenarios,
            strategies=["DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"],
            status="completed",
        ),
    ]
