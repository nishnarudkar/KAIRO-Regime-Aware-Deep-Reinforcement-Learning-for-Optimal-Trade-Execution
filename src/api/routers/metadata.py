"""
src/api/routers/metadata.py — Platform Metadata Routes

Endpoints:
  GET /api/baselines   — List available baseline strategies (TWAP, VWAP, POV)
  GET /api/models      — List learned policies with their real training / evaluation status
  GET /api/experiments — List research experiment suites that have result files on disk
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from src.agents import registry
import numpy as np
import pandas as pd

from src.api.schemas import (
    ExperimentResultsResponse,
    BaselineStrategyResponse,
    ModelMetadataResponse,
    ExperimentSummaryResponse,
)
from src.evaluation.scenarios import SCENARIOS

router = APIRouter(tags=["Platform Metadata"])


def results_dir() -> Path:
    return Path(os.environ.get("KAIRO_RESULTS_DIR", registry.REPO_ROOT / "results"))


@router.get("/api/baselines", response_model=List[BaselineStrategyResponse])
def list_baselines():
    """List all available conventional execution baseline strategies."""
    return [
        BaselineStrategyResponse(
            strategy_id="TWAP",
            name="Time-Weighted Average Price",
            description="Executes inventory uniformly across horizon steps: v_t = remaining / steps_left",
            parameters={"target_inventory": 100000.0, "total_steps": 30},
        ),
        BaselineStrategyResponse(
            strategy_id="VWAP",
            name="Volume-Weighted Average Price",
            description=(
                "Executes inventory in proportion to an ex-ante intraday volume profile estimated "
                "from earlier trading days (minute-of-day averages); flat (= TWAP) when no history exists."
            ),
            parameters={"target_inventory": 100000.0, "total_steps": 30},
        ),
        BaselineStrategyResponse(
            strategy_id="POV",
            name="Percentage of Volume",
            description="Executes a fixed percentage (10%) of prevailing market volume at each step: v_t = 0.10 * V_t",
            parameters={"target_inventory": 100000.0, "target_rate": 0.10},
        ),
    ]


@router.get("/api/models", response_model=List[ModelMetadataResponse])
def list_models():
    """List learned policies. ``status`` is 'trained' only when a checkpoint exists on disk."""
    fields = ModelMetadataResponse.model_fields
    return [ModelMetadataResponse(**{k: m[k] for k in fields}) for m in registry.describe_models()]


def _records(path: Path) -> List[dict]:
    """CSV -> list of dicts with NaN converted to None (valid JSON)."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")


@router.get("/api/experiments/results", response_model=ExperimentResultsResponse)
def experiment_results():
    """Recorded research results: per-strategy summary, paired comparisons and HMM validation."""
    rdir = results_dir()
    if not (rdir / "window_results.csv").exists():
        raise HTTPException(status_code=404, detail="No experiment results found. Run scripts/run_experiments.py.")
    cfg = {}
    try:
        cfg = json.loads((rdir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    hmm = {}
    hpath = rdir / "hmm_validation.csv"
    if hpath.exists():
        h = pd.read_csv(hpath)
        hmm = {"mean_ari": float(np.nanmean(h["test_ari"])) if "test_ari" in h else None,
               "mean_accuracy": float(np.nanmean(h["test_accuracy"])) if "test_accuracy" in h else None,
               "n_runs": int(len(h))}
    return ExperimentResultsResponse(
        config=cfg,
        summary=_records(rdir / "summary_table.csv"),
        comparisons=_records(rdir / "paired_comparisons.csv"),
        hmm_validation=hmm,
    )


@router.get("/api/experiments", response_model=List[ExperimentSummaryResponse])
def list_experiments():
    """List research experiment suites; 'completed' only if their result files exist."""
    rdir = results_dir()
    cfg_path = rdir / "run_config.json"
    have_results = (rdir / "window_results.csv").exists() and cfg_path.exists()
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
    strategies = ["TWAP", "VWAP", "POV", "DQN", "Regime-Aware DQN", "DQN (shuffled regime)",
                  "PPO", "Regime-Aware PPO", "PPO (shuffled regime)"]
    return [
        ExperimentSummaryResponse(
            experiment_id="research_suite",
            name="Research experiment suite (RQ1-RQ4)",
            scenarios=cfg.get("scenarios", list(SCENARIOS.keys())),
            strategies=strategies,
            status="completed" if have_results else "not_run",
            results_available=have_results,
            seeds=cfg.get("seeds"),
            train_timesteps=cfg.get("train_timesteps"),
        ),
    ]
