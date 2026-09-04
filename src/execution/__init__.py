"""
Execution engines, market impact models (Almgren-Chriss), and transaction cost accounting.
"""

from abc import ABC, abstractmethod

class BaseImpactModel(ABC):
    """Abstract interface for temporary and permanent market impact calculation."""

    @abstractmethod
    def calculate_impact(self, trade_size: float, volume: float, volatility: float) -> float:
        """Calculate price impact for a given trade action."""
        pass
