"""
Data pipelines, loaders, and feature engineering interfaces.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    """Abstract interface for loading market data."""
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load market data from disk or database."""
        pass

class BaseFeatureExtractor(ABC):
    """Abstract interface for causal feature engineering."""

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical, microstructural, and volatility features without lookahead."""
        pass
