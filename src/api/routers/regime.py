"""
src/api/routers/regime.py — Market Regime Detection Route

Endpoints:
  GET /api/regime/current  — Detect the regime at the latest bar of a synthetic market
"""

from __future__ import annotations

import datetime
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from src.agents import registry
from src.api import engine
from src.api.schemas import CurrentRegimeResponse
from src.evaluation import protocol as P
from src.evaluation.scenarios import REGIME_NAMES, SCENARIOS
from src.regimes.inference import CausalRegimeInference

router = APIRouter(prefix="/api/regime", tags=["Market Regimes"])

REGIME_LABELS = {i: n for i, n in enumerate(REGIME_NAMES)}


@lru_cache(maxsize=8)
def _fallback_hmm(scenario: str, seed: int):
    """HMM fitted on the train region of this market, used only when no pooled model is saved."""
    _, split = engine.get_market(scenario, seed)
    return P.fit_hmm([split.train])


@router.get("/current", response_model=CurrentRegimeResponse, status_code=status.HTTP_200_OK)
def get_current_regime(
    symbol: str = Query(default="AAPL", description="Ticker symbol"),
    scenario: str = Query(default="normal", description="Market scenario to sample"),
    seed: int = Query(default=42, description="Random seed"),
):
    """
    Causal HMM regime at the latest bar of a synthetic market.

    Uses the pooled HMM saved by ``scripts/train_models.py`` when available (no fitting per request)
    and runs the forward filter over the whole history: P(S_t | X_1..t), no smoothing, no lookahead.
    """
    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown scenario '{scenario}'. Valid options: {list(SCENARIOS.keys())}",
        )

    df, _ = engine.get_market(scenario, seed)
    hmm = registry.load_hmm() or _fallback_hmm(scenario, seed)
    if hmm is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Regime model could not be fitted.")

    feats = P.compute_regime_features(df)
    X = feats[hmm.feature_names].dropna().to_numpy()
    infer = CausalRegimeInference(hmm)
    infer.reset_online_state()
    regime_id, probs = 0, np.ones(hmm.n_regimes) / hmm.n_regimes
    for x in X:
        regime_id, probs = infer.step_online(x)

    last = df.iloc[-1]
    return CurrentRegimeResponse(
        symbol=symbol,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        current_price=float(last["price"]),
        spread=float(last["spread"]),
        volatility=float(last["volatility"]),
        regime_id=int(regime_id),
        regime_label=REGIME_LABELS.get(int(regime_id), f"Regime_{regime_id}"),
        regime_probabilities={REGIME_LABELS.get(i, f"Regime_{i}"): float(p) for i, p in enumerate(probs)},
        true_regime=REGIME_LABELS.get(int(last["true_regime"])),
    )
