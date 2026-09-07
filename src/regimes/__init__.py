"""
Regime Detection Engine module interfaces, features, HMM model, causal inference, and evaluation.
"""

from src.regimes.features import (
    RegimeFeatureEngine,
    RegimeFeatureScaler,
    chronological_split,
    DEFAULT_REGIME_FEATURES,
)
from src.regimes.hmm_model import MarketHMM, CANONICAL_REGIME_LABELS
from src.regimes.inference import CausalRegimeInference
from src.regimes.evaluation import RegimeEvaluator

__all__ = [
    "RegimeFeatureEngine",
    "RegimeFeatureScaler",
    "chronological_split",
    "DEFAULT_REGIME_FEATURES",
    "MarketHMM",
    "CANONICAL_REGIME_LABELS",
    "CausalRegimeInference",
    "RegimeEvaluator",
]
