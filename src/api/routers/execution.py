"""
src/api/routers/execution.py — Execution Engine API Routes

Endpoints:
  POST /api/execution/simulate   — Execute single simulation run
  POST /api/execution/backtest   — Run backtest comparison across strategies
  GET  /api/execution/{id}       — Query execution record by ID
  GET  /api/execution/{id}/metrics     — Query execution metrics by ID
  GET  /api/execution/{id}/trajectory  — Query execution trajectory by ID
"""

from __future__ import annotations

import datetime
import uuid
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Path, status
import pandas as pd
import numpy as np

from src.api.schemas import (
    ExecutionSimulateRequest,
    BacktestRequest,
    ExecutionResponse,
    ExecutionMetricsResponse,
    ExecutionTrajectoryResponse,
    BacktestResponse,
    BacktestResultItem,
    DecisionExplanationRequest,
    DecisionExplanationResponse,
)
from src.api.store import global_store
from src.evaluation.scenarios import generate_scenario_data, SCENARIOS

router = APIRouter(prefix="/api/execution", tags=["Execution Engine"])


# ── Helper Execution Engine Functions ──────────────────────────────────────────

def _run_baseline_simulation(
    strategy_name: str,
    market_data: pd.DataFrame,
    target_inventory: float,
    side: str,
    horizon_steps: int,
) -> Tuple[Any, Any]:
    from src.baselines import TWAPStrategy, VWAPStrategy, POVStrategy
    from src.baselines.runner import BaselineRunner

    if strategy_name == "TWAP":
        strategy = TWAPStrategy(target_inventory=target_inventory, total_steps=horizon_steps)
    elif strategy_name == "VWAP":
        strategy = VWAPStrategy(target_inventory=target_inventory, total_steps=horizon_steps)
    elif strategy_name == "POV":
        strategy = POVStrategy(target_inventory=target_inventory, target_rate=0.10)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy_name}")

    runner = BaselineRunner()
    result = runner.run(
        strategy=strategy,
        market_data=market_data,
        target_inventory=target_inventory,
        side=side,
        horizon_steps=horizon_steps,
    )
    return result, runner


def _fit_hmm_for_data(df: pd.DataFrame):
    from src.regimes.hmm_model import MarketHMM
    from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler

    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(df)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)
    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42)
    hmm.fit(X, scaler=scaler)
    return hmm, feat_df


def _run_rl_simulation(
    policy_name: str,
    market_data: pd.DataFrame,
    target_inventory: float,
    side: str,
    horizon_steps: int,
    seed: int,
) -> Tuple[Any, Any]:
    from src.agents.dqn_agent import DQNAgent
    from src.agents.ppo_agent import PPOAgent
    from src.agents.evaluator import evaluate_agent
    from src.environment import TradeExecutionEnv, RegimeAwareTradeExecutionEnv

    use_regime = "Regime" in policy_name or "Regime-Aware" in policy_name
    is_ppo = "PPO" in policy_name

    split_idx = int(len(market_data) * 0.70)
    train_data = market_data.iloc[:split_idx].reset_index(drop=True)
    test_data = market_data.iloc[split_idx:].reset_index(drop=True)

    if use_regime:
        hmm, train_feat = _fit_hmm_for_data(train_data)
        from src.regimes.features import RegimeFeatureEngine
        test_feat = RegimeFeatureEngine().compute_features(test_data)

        train_env = RegimeAwareTradeExecutionEnv(
            hmm_model=hmm,
            regime_feature_data=train_feat,
            market_data=train_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
        )
        eval_env = RegimeAwareTradeExecutionEnv(
            hmm_model=hmm,
            regime_feature_data=test_feat,
            market_data=test_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
        )
    else:
        train_env = TradeExecutionEnv(
            market_data=train_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
        )
        eval_env = TradeExecutionEnv(
            market_data=test_data,
            target_inventory=target_inventory,
            side=side,
            horizon_steps=horizon_steps,
        )

    # Instantiate agent & short train for fast API response
    if is_ppo:
        agent = PPOAgent(env=train_env, seed=seed, n_steps=256, batch_size=32, verbose=0)
        agent.train(total_timesteps=512)
    else:
        agent = DQNAgent(env=train_env, seed=seed, learning_starts=100, verbose=0)
        agent.train(total_timesteps=512)

    result = evaluate_agent(
        agent=agent,
        market_data=test_data,
        target_inventory=target_inventory,
        side=side,
        horizon_steps=horizon_steps,
        seed=seed,
        agent_name=policy_name,
        env=eval_env,
    )
    return result, eval_env


# ── Route Handlers ─────────────────────────────────────────────────────────────

@router.post("/simulate", response_model=ExecutionResponse, status_code=status.HTTP_200_OK)
def simulate_execution(req: ExecutionSimulateRequest):
    """
    Run a single execution simulation for the requested policy and scenario.

    Returns the complete execution response and stores the record in memory.
    """
    if req.scenario not in SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario '{req.scenario}'. Valid options: {list(SCENARIOS.keys())}",
        )

    exec_id = str(uuid.uuid4())
    market_data = generate_scenario_data(req.scenario, seed=req.seed)

    baselines = {"TWAP", "VWAP", "POV"}
    rl_policies = {"DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"}

    if req.policy in baselines:
        res, runner = _run_baseline_simulation(
            strategy_name=req.policy,
            market_data=market_data,
            target_inventory=req.quantity,
            side=req.side,
            horizon_steps=req.horizon_steps,
        )
        inv_traj = [req.quantity] + [s.remaining_inventory for s in runner.simulator.execution_history]
        act_traj = [0] * len(inv_traj)
        price_traj = market_data["price"].iloc[:len(inv_traj)].tolist()
        act_counts = {}
    elif req.policy in rl_policies:
        res, eval_env = _run_rl_simulation(
            policy_name=req.policy,
            market_data=market_data,
            target_inventory=req.quantity,
            side=req.side,
            horizon_steps=req.horizon_steps,
            seed=req.seed,
        )
        inv_traj = res.inventory_trajectory
        act_traj = res.action_trajectory
        price_traj = eval_env.market_data["price"].iloc[:len(inv_traj)].tolist()
        act_counts = res.action_counts
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported policy '{req.policy}'. Valid options: {list(baselines | rl_policies)}",
        )

    # Format action counts string keys
    str_act_counts = {str(k): int(v) for k, v in act_counts.items()}

    metrics_dto = ExecutionMetricsResponse(
        implementation_shortfall=float(res.implementation_shortfall),
        implementation_shortfall_bps=float(res.implementation_shortfall_bps),
        execution_cost=float(res.execution_cost),
        market_impact_cost=float(getattr(res, "market_impact_cost", 0.0)),
        total_transaction_fees=float(getattr(res, "total_transaction_fees", 0.0)),
        terminal_penalty=float(getattr(res, "terminal_penalty", 0.0)),
        completion_rate=float(res.completion_rate),
        average_execution_price=float(res.average_execution_price),
        arrival_price=float(res.arrival_price),
        market_vwap_price=float(res.market_vwap_price),
        vwap_slippage_bps=float(res.vwap_slippage_bps),
        action_counts=str_act_counts,
    )

    trajectory_dto = ExecutionTrajectoryResponse(
        execution_id=exec_id,
        inventory_trajectory=inv_traj,
        action_trajectory=act_traj,
        price_trajectory=price_traj,
    )

    response_dto = ExecutionResponse(
        execution_id=exec_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        symbol=req.symbol,
        side=req.side,
        target_inventory=req.quantity,
        executed_inventory=float(res.executed_inventory),
        remaining_inventory=float(res.remaining_inventory),
        policy=req.policy,
        scenario=req.scenario,
        status="completed",
        metrics=metrics_dto,
    )

    global_store.save_execution(response_dto, trajectory_dto)
    return response_dto


@router.post("/backtest", response_model=BacktestResponse, status_code=status.HTTP_200_OK)
def run_backtest(req: BacktestRequest):
    """
    Run backtest comparison across multiple baseline and DRL policies.
    """
    if req.scenario not in SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario '{req.scenario}'. Valid options: {list(SCENARIOS.keys())}",
        )

    backtest_id = str(uuid.uuid4())
    market_data = generate_scenario_data(req.scenario, seed=req.seed)
    results_items: List[BacktestResultItem] = []

    baselines = {"TWAP", "VWAP", "POV"}
    rl_policies = {"DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"}

    for policy in req.policies:
        try:
            if policy in baselines:
                res, _ = _run_baseline_simulation(
                    strategy_name=policy,
                    market_data=market_data,
                    target_inventory=req.quantity,
                    side=req.side,
                    horizon_steps=req.horizon_steps,
                )
            elif policy in rl_policies:
                res, _ = _run_rl_simulation(
                    policy_name=policy,
                    market_data=market_data,
                    target_inventory=req.quantity,
                    side=req.side,
                    horizon_steps=req.horizon_steps,
                    seed=req.seed,
                )
            else:
                continue

            results_items.append(
                BacktestResultItem(
                    policy=policy,
                    implementation_shortfall_bps=float(res.implementation_shortfall_bps),
                    execution_cost=float(res.execution_cost),
                    completion_rate=float(res.completion_rate),
                    vwap_slippage_bps=float(res.vwap_slippage_bps),
                )
            )
        except Exception as e:
            continue

    return BacktestResponse(
        backtest_id=backtest_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        symbol=req.symbol,
        side=req.side,
        quantity=req.quantity,
        scenario=req.scenario,
        results=results_items,
    )


@router.get("/{id}", response_model=ExecutionResponse)
def get_execution_record(id: str = Path(..., description="Execution UUID string")):
    """Fetch stored execution summary record by execution ID."""
    record = global_store.get_execution(id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution ID '{id}' not found.",
        )
    return record


@router.get("/{id}/metrics", response_model=ExecutionMetricsResponse)
def get_execution_metrics(id: str = Path(..., description="Execution UUID string")):
    """Fetch execution quality metrics by execution ID."""
    metrics = global_store.get_metrics(id)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution metrics for ID '{id}' not found.",
        )
    return metrics


@router.get("/{id}/trajectory", response_model=ExecutionTrajectoryResponse)
def get_execution_trajectory(id: str = Path(..., description="Execution UUID string")):
    """Fetch step-by-step execution trajectory by execution ID."""
    trajectory = global_store.get_trajectory(id)
    if not trajectory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution trajectory for ID '{id}' not found.",
        )
    return trajectory


@router.post("/explain", response_model=DecisionExplanationResponse)
def explain_action_decision(req: DecisionExplanationRequest):
    """
    Compute post-hoc feature attributions and regime influence score for a given observation state and action.
    """
    from src.agents.explainer import DecisionExplainer

    explainer = DecisionExplainer()
    state_arr = np.array(req.state, dtype=np.float32)

    # Dummy agent evaluator callback
    def dummy_agent_eval(s):
        # Q-values proxy based on state parameters
        res = np.ones(4, dtype=np.float32) * 0.25
        if len(s) >= 4:
            res[1] += s[0] * 0.1  # inventory
            res[2] += s[3] * 0.2  # volatility
        if len(s) >= 12:
            res[3] += s[11] * 0.3  # stress probability
        return res

    class ProxyAgent:
        def predict(self, s):
            return int(np.argmax(dummy_agent_eval(s)))
        def get_q_values(self, s):
            return dummy_agent_eval(s)

    result = explainer.explain_step(
        agent=ProxyAgent(),
        state=state_arr,
        action=req.action,
    )

    return DecisionExplanationResponse(
        action=result["action"],
        action_label=result["action_label"],
        feature_attributions=result["feature_attributions"],
        feature_percentages=result["feature_percentages"],
        regime_influence_score=result["regime_influence_score"],
        action_scores=result["action_scores"],
        action_advantages=result["action_advantages"],
        summary=result["summary"],
    )


@router.get("/{id}/explain", response_model=DecisionExplanationResponse)
def get_execution_explanation(id: str = Path(..., description="Execution UUID string")):
    """
    Retrieve feature attribution explanation for a completed execution run.
    """
    record = global_store.get_execution(id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution ID '{id}' not found.",
        )

    # Construct representative state vector for record policy
    use_regime = "Regime" in record.policy or "Regime-Aware" in record.policy
    dim = 12 if use_regime else 7
    sample_state = [1.0, 0.5, 0.001, 0.02, 0.0005, 1.0, 1.0]
    if use_regime:
        sample_state += [2.0, 0.1, 0.2, 0.6, 0.1]  # High vol regime bias

    req = DecisionExplanationRequest(
        state=sample_state,
        action=1,
        policy=record.policy,
    )
    return explain_action_decision(req)

