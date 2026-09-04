"""
Regime Detection Engine module interfaces (HMM & Volatility classifiers).
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseRegimeDetector(ABC):
    """Abstract interface for market regime detection."""

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> None:
        """Fit regime model on historical training features."""
        pass

    @abstractmethod
    def predict_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Predict discrete market regime IDs (0..K-1)."""
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict soft regime posterior probability distribution."""
        pass
