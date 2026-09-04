"""
Feature engineering pipelines and market indicators (volatility, spread, momentum, OFI).
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseFeatureCalculator(ABC):
    """Abstract interface for microstructural and statistical feature calculation."""

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features without future lookahead bias."""
        pass
