"""
src/api/routers/execution.py — Execution Engine API Routes

Endpoints:
  POST /api/execution/simulate   — Execute single simulation run
  POST /api/execution/backtest   — Run backtest comparison across strategies
  POST /api/execution/paper      — Validate order against risk gates & execute paper slice
  GET  /api/execution/risk-status — Query pre-trade risk gate parameters & kill switch status
  POST /api/execution/kill-switch — Trigger / reset emergency halt kill switch
  POST /api/execution/explain    — Compute post-hoc feature attributions & regime influence score
  GET  /api/execution/{id}       — Query execution record by ID
  GET  /api/execution/{id}/metrics     — Query execution metrics by ID
  GET  /api/execution/{id}/trajectory  — Query execution trajectory by ID
  GET  /api/execution/{id}/explain     — Query decision explanation for completed run
"""

from __future__ import annotations

import datetime
import uuid
from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
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
    BacktestError,
    DecisionExplanationRequest,
    DecisionExplanationResponse,
    PaperOrderRequest,
    PaperOrderResponse,
    RiskStatusResponse,
    SetKillSwitchRequest,
)
from src.api.store import global_store
from src.api.security import require_api_key
from src.api import engine
from src.agents import registry
from src.evaluation.scenarios import generate_scenario_data, SCENARIOS
from src.execution.risk_gates import ExecutionRiskGate, RiskGateConfig
from src.execution.alpaca_paper import AlpacaPaperExecutor

router = APIRouter(prefix="/api/execution", tags=["Execution Engine"])

# Global Risk Gate & Paper Executor Instances
global_risk_gate = ExecutionRiskGate()
global_paper_executor = AlpacaPaperExecutor(risk_gate=global_risk_gate, mock_mode=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

BASELINE_SET = set(registry.BASELINE_POLICIES)
RL_SET = set(registry.RL_POLICIES)
ALL_POLICIES = sorted(BASELINE_SET | RL_SET)


def _check_scenario(name: str) -> None:
    if name not in SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario '{name}'. Valid options: {list(SCENARIOS.keys())}",
        )


def _check_side(side: str) -> str:
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="side must be 'BUY' or 'SELL'.")
    return side


def _metrics_dto(r: Dict[str, Any], counts: Dict[str, int]) -> ExecutionMetricsResponse:
    return ExecutionMetricsResponse(
        implementation_shortfall=float(r["implementation_shortfall"]),
        implementation_shortfall_bps=float(r["implementation_shortfall_bps"]),
        execution_cost=float(r["total_execution_cost"]),
        market_impact_cost=float(r["total_impact_cost"]),
        total_transaction_fees=float(r["total_transaction_fees"]),
        terminal_penalty=float(r["terminal_penalty"]),
        completion_rate=float(min(1.0, r["fill_rate"])),
        average_execution_price=float(r["average_execution_price"]),
        arrival_price=float(r["arrival_price"]),
        market_vwap_price=float(r["vwap_market_price"]),
        vwap_slippage_bps=float(r["vwap_slippage_bps"]),
        action_counts=counts,
    )


# ── Static Routes (MUST be defined before /{id} parameter routes) ────────────────

@router.post("/simulate", response_model=ExecutionResponse, status_code=status.HTTP_200_OK)
def simulate_execution(req: ExecutionSimulateRequest):
    """
    Run one execution on an out-of-sample window of a synthetic market.

    Learned policies are served from trained checkpoints (see scripts/train_models.py);
    a policy without a checkpoint returns 503 instead of being trained on the fly.
    """
    _check_scenario(req.scenario)
    side = _check_side(req.side)
    if req.policy not in BASELINE_SET | RL_SET:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported policy '{req.policy}'. Valid options: {ALL_POLICIES}",
        )

    df, split = engine.get_market(req.scenario, req.seed)
    starts = engine.test_starts(split, req.horizon_steps)
    if not starts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="horizon_steps too large for the test region.")
    start = engine.pick_window(starts, req.seed)

    try:
        res = engine.run_policy(req.policy, df, start, req.horizon_steps, side, req.quantity, record_obs=True)
    except registry.ModelUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    exec_id = str(uuid.uuid4())
    counts = engine.action_counts(res["action_trajectory"])
    metrics_dto = _metrics_dto(res, counts)

    trajectory_dto = ExecutionTrajectoryResponse(
        execution_id=exec_id,
        inventory_trajectory=res["inventory_trajectory"],
        action_trajectory=res["action_trajectory"],
        price_trajectory=res["price_trajectory"],
        regime_trajectory=res["regime_trajectory"] or None,
        observation_trajectory=res["observation_trajectory"] or None,
        window_start=start,
    )
    response_dto = ExecutionResponse(
        execution_id=exec_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        symbol=req.symbol,
        side=side,
        target_inventory=req.quantity,
        executed_inventory=float(res["executed_inventory"]),
        remaining_inventory=float(res["remaining_inventory"]),
        policy=req.policy,
        scenario=req.scenario,
        status="completed" if metrics_dto.completion_rate >= 0.9999 else "partial",
        metrics=metrics_dto,
    )
    global_store.save_execution(response_dto, trajectory_dto)
    return response_dto


@router.post("/backtest", response_model=BacktestResponse, status_code=status.HTTP_200_OK)
def run_backtest(req: BacktestRequest):
    """
    Compare policies over several out-of-sample windows of one synthetic market.

    Every policy runs on exactly the same windows (paired comparison). Policies that cannot
    be evaluated are listed in ``errors`` instead of being silently dropped.
    """
    _check_scenario(req.scenario)
    side = _check_side(req.side)
    unknown = [p for p in req.policies if p not in BASELINE_SET | RL_SET]
    if unknown:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported policies {unknown}. Valid options: {ALL_POLICIES}")

    df, split = engine.get_market(req.scenario, req.seed)
    starts = engine.test_starts(split, req.horizon_steps)
    if not starts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="horizon_steps too large for the test region.")
    offset = req.seed % len(starts)
    starts = (starts[offset:] + starts[:offset])[: req.n_windows]

    per_policy: Dict[str, Dict[str, np.ndarray]] = {}
    errors: List[BacktestError] = []
    policies = list(dict.fromkeys(req.policies))
    if "TWAP" not in policies:
        policies.append("TWAP")   # reference for the paired difference
    for policy in policies:
        try:
            runs = [engine.run_policy(policy, df, s, req.horizon_steps, side, req.quantity) for s in starts]
        except registry.ModelUnavailable as exc:
            errors.append(BacktestError(policy=policy, detail=str(exc)))
            continue
        except Exception as exc:
            errors.append(BacktestError(policy=policy, detail=f"{type(exc).__name__}: {exc}"))
            continue
        per_policy[policy] = {
            "is": np.array([r["implementation_shortfall_bps"] for r in runs]),
            "cost": np.array([r["total_execution_cost"] for r in runs]),
            "fill": np.array([min(1.0, r["fill_rate"]) for r in runs]),
            "slip": np.array([r["vwap_slippage_bps"] for r in runs]),
        }

    twap = per_policy.get("TWAP")
    items: List[BacktestResultItem] = []
    for policy in req.policies:
        d = per_policy.get(policy)
        if d is None:
            continue
        lo, hi = engine.bootstrap_ci(d["is"])
        items.append(BacktestResultItem(
            policy=policy,
            implementation_shortfall_bps=float(d["is"].mean()),
            implementation_shortfall_bps_std=float(d["is"].std(ddof=1)) if len(d["is"]) > 1 else 0.0,
            ci_low=lo, ci_high=hi,
            vs_twap_bps=float((d["is"] - twap["is"]).mean()) if twap is not None and policy != "TWAP" else None,
            vs_twap_ci_low=(engine.bootstrap_ci(d["is"] - twap["is"])[0] if twap is not None and policy != "TWAP" else None),
            vs_twap_ci_high=(engine.bootstrap_ci(d["is"] - twap["is"])[1] if twap is not None and policy != "TWAP" else None),
            execution_cost=float(d["cost"].mean()),
            completion_rate=float(d["fill"].mean()),
            vwap_slippage_bps=float(d["slip"].mean()),
            n_windows=int(len(starts)),
        ))

    return BacktestResponse(
        backtest_id=str(uuid.uuid4()),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        symbol=req.symbol,
        side=side,
        quantity=req.quantity,
        scenario=req.scenario,
        n_windows=len(starts),
        results=items,
        errors=[e for e in errors if e.policy in req.policies],
    )


@router.post("/paper", response_model=PaperOrderResponse, status_code=status.HTTP_200_OK,
             dependencies=[Depends(require_api_key)])
def submit_paper_order(req: PaperOrderRequest):
    """Validate order against pre-trade risk gates and route to Alpaca Paper Trading execution engine."""
    result = global_paper_executor.execute_slice(
        symbol=req.symbol,
        side=req.side,
        quantity=req.quantity,
        current_price=req.current_price,
        arrival_price=req.arrival_price,
        target_inventory=req.target_inventory,
    )

    return PaperOrderResponse(
        order_id=result.get("order_id"),
        status=result["status"],
        mode=result.get("mode", "paper_mock"),
        symbol=req.symbol,
        side=req.side,
        executed_quantity=result["executed_quantity"],
        fill_price=result["fill_price"],
        reason=result["reason"],
        timestamp=result["timestamp"],
    )


@router.get("/risk-status", response_model=RiskStatusResponse)
def get_risk_status():
    """Query active pre-trade risk gate configuration limits and emergency kill switch status."""
    st = global_risk_gate.get_status()
    return RiskStatusResponse(
        max_notional_value=st["max_notional_value"],
        max_single_order_pct=st["max_single_order_pct"],
        price_collar_pct=st["price_collar_pct"],
        kill_switch_active=st["kill_switch_active"],
    )


@router.post("/kill-switch", response_model=RiskStatusResponse, dependencies=[Depends(require_api_key)])
def set_kill_switch(req: SetKillSwitchRequest):
    """Trigger or reset emergency kill switch to immediately halt paper execution."""
    global_risk_gate.set_kill_switch(req.active)
    return get_risk_status()


def _explain(policy: str, state: List[float], action: Optional[int]) -> DecisionExplanationResponse:
    """Explain a decision of a *trained* policy network (finite-difference sensitivities)."""
    from src.agents.explainer import DecisionExplainer

    if policy not in RL_SET:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Only learned policies can be explained: {sorted(RL_SET)}")
    try:
        lp = registry.load_policy(policy)
    except registry.ModelUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if len(state) != lp.spec["state_dim"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"'{policy}' expects a {lp.spec['state_dim']}-dimensional state, got {len(state)}.")

    state_arr = np.array(state, dtype=np.float32)
    chosen = int(action) if action is not None else int(lp.agent.predict(state_arr, deterministic=True))
    result = DecisionExplainer().explain_step(agent=lp.agent, state=state_arr, action=chosen)
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


@router.post("/explain", response_model=DecisionExplanationResponse)
def explain_action_decision(req: DecisionExplanationRequest):
    """Feature attributions and regime influence for a decision of the trained policy network."""
    return _explain(req.policy, req.state, req.action)


# ── Dynamic Parameter Routes (MUST be defined after static routes) ──────────────

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


@router.get("/{id}/explain", response_model=DecisionExplanationResponse)
def get_execution_explanation(
    id: str = Path(..., description="Execution UUID string"),
    step: int = Query(default=0, ge=0, description="Step of the execution to explain"),
):
    """Retrieve feature attribution explanation for a completed execution run."""
    record = global_store.get_execution(id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution ID '{id}' not found.",
        )

    traj = global_store.get_trajectory(id)
    if record.policy not in RL_SET or not traj or not traj.observation_trajectory:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explanations are only available for learned policies (baselines have no network).",
        )
    obs = traj.observation_trajectory
    if step >= len(obs):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"step must be < {len(obs)} for this execution.")
    return _explain(record.policy, obs[step], traj.action_trajectory[step])
