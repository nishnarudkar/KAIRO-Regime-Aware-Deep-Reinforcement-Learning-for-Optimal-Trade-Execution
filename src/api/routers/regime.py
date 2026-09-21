"""
src/api/routers/regime.py — Market Regime Detection Route

Endpoints:
  GET /api/regime/current  — Detect and return current market regime
"""

from __future__ import annotations

import datetime
from typing import Dict, Any

from fastapi import APIRouter, Query, HTTPException, status
import pandas as pd
import numpy as np

from src.api.schemas import CurrentRegimeResponse
from src.evaluation.scenarios import generate_scenario_data
from src.regimes.hmm_model import MarketHMM
from src.regimes.features import RegimeFeatureEngine, RegimeFeatureScaler
from src.regimes.inference import CausalRegimeInference

router = APIRouter(prefix="/api/regime", tags=["Market Regimes"])

REGIME_LABELS = {
    0: "Low Volatility",
    1: "Normal",
    2: "High Volatility",
    3: "Stress",
}


@router.get("/current", response_model=CurrentRegimeResponse, status_code=status.HTTP_200_OK)
def get_current_regime(
    symbol: str = Query(default="AAPL", description="Ticker symbol"),
    scenario: str = Query(default="normal", description="Market scenario to sample"),
    seed: int = Query(default=42, description="Random seed"),
):
    """
    Detect current market regime using the causal HMM forward algorithm.

    Returns the latest market metrics (price, spread, volatility), estimated regime ID,
    human-readable label, and posterior regime probabilities P(S_t = k | X_{1:t}).
    """
    try:
        market_data = generate_scenario_data(scenario, seed=seed)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    engine = RegimeFeatureEngine()
    feat_df = engine.compute_features(market_data)
    _, X = engine.extract_feature_matrix(feat_df, drop_na=True)

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42)
    hmm.fit(X, scaler=scaler)

    # Causal forward inference online step
    infer_engine = CausalRegimeInference(hmm)
    regime_id = 0
    probs = np.ones(4) / 4.0

    for i in range(len(X)):
        regime_id, probs = infer_engine.step_online(X[i])

    latest_row = market_data.iloc[-1]
    regime_probs = {
        REGIME_LABELS.get(i, f"Regime_{i}"): float(probs[i])
        for i in range(len(probs))
    }

    return CurrentRegimeResponse(
        symbol=symbol,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        current_price=float(latest_row["price"]),
        spread=float(latest_row["spread"]),
        volatility=float(latest_row["volatility"]),
        regime_id=int(regime_id),
        regime_label=REGIME_LABELS.get(int(regime_id), f"Regime_{regime_id}"),
        regime_probabilities=regime_probs,
    )
