"""
Evaluation, metrics calculation, and backtesting suite.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseEvaluator(ABC):
    """Abstract interface for backtesting and evaluating execution strategies."""

    @abstractmethod
    def evaluate(self, strategy_results: pd.DataFrame) -> dict:
        """Calculate Implementation Shortfall, VWAP slippage, and fill rates."""
        pass
