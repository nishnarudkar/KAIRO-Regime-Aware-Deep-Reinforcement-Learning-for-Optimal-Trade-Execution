"""
Market Impact Models for Trade Execution Simulation.

Supports linear and Almgren-Chriss power-law temporary and permanent market impact models.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseImpactModel(ABC):
    """Abstract interface for temporary and permanent market impact calculation."""

    @abstractmethod
    def calculate_impact(
        self,
        trade_size: float,
        volume: float,
        volatility: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        """
        Calculate temporary price impact ($/share) for a given execution trade action.
        
        Args:
            trade_size: Executed trade size in shares.
            volume: Market bar volume in shares.
            volatility: Instantaneous volatility.
            price: Baseline market mid price.
            side: 'BUY' or 'SELL'.
            
        Returns:
            Temporary impact in currency units per share ($/share addition to execution price).
        """
        pass

    @abstractmethod
    def calculate_permanent_impact(
        self,
        trade_size: float,
        volume: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        """
        Calculate permanent price impact ($/share shift in mid price) caused by execution.
        
        Returns:
            Permanent price shift in currency units ($/share shift in underlying mid price).
        """
        pass


class LinearImpactModel(BaseImpactModel):
    """
    Simple parameterized linear impact model.
    
    Temporary Impact:  temp_impact = sign * eta * (trade_size / volume) * price
    Permanent Impact:  perm_impact = sign * gamma * (trade_size / volume) * price
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.05):
        """
        Args:
            eta: Temporary impact linear coefficient.
            gamma: Permanent impact linear coefficient.
        """
        self.eta = eta
        self.gamma = gamma

    def calculate_impact(
        self,
        trade_size: float,
        volume: float,
        volatility: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        if trade_size <= 0 or volume <= 0:
            return 0.0
        
        sign = 1.0 if side.upper() == "BUY" else -1.0
        participation_rate = trade_size / volume
        temp_impact = sign * self.eta * participation_rate * price
        return temp_impact

    def calculate_permanent_impact(
        self,
        trade_size: float,
        volume: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        if trade_size <= 0 or volume <= 0:
            return 0.0

        sign = 1.0 if side.upper() == "BUY" else -1.0
        participation_rate = trade_size / volume
        perm_impact = sign * self.gamma * participation_rate * price
        return perm_impact


class AlmgrenChrissImpactModel(BaseImpactModel):
    """
    Almgren-Chriss (2000) square-root / power-law temporary and linear permanent impact model.
    
    Temporary Impact:  temp_impact = sign * eta * (trade_size / volume)^alpha * volatility * price
    Permanent Impact:  perm_impact = sign * gamma * (trade_size / volume) * price
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.01, alpha: float = 0.5):
        """
        Args:
            eta: Temporary impact scaling coefficient.
            gamma: Permanent impact linear coefficient.
            alpha: Temporary impact volume exponent (default: 0.5 for square-root law).
        """
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha

    def calculate_impact(
        self,
        trade_size: float,
        volume: float,
        volatility: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        if trade_size <= 0 or volume <= 0:
            return 0.0

        sign = 1.0 if side.upper() == "BUY" else -1.0
        participation_rate = trade_size / volume
        # Protect against negative or NaN volatility
        vol = max(volatility, 1e-6)
        temp_impact = sign * self.eta * (participation_rate ** self.alpha) * vol * price
        return temp_impact

    def calculate_permanent_impact(
        self,
        trade_size: float,
        volume: float,
        price: float,
        side: str = "BUY"
    ) -> float:
        if trade_size <= 0 or volume <= 0:
            return 0.0

        sign = 1.0 if side.upper() == "BUY" else -1.0
        participation_rate = trade_size / volume
        perm_impact = sign * self.gamma * participation_rate * price
        return perm_impact


# Calibration used by every environment / runner unless a model is passed explicitly.
#   eta   = 0.5    square-root law temporary impact  eta * sigma_bar * sqrt(q / V) * P
#   gamma = 0.001  small linear permanent impact     gamma * (q / V) * P
# For a 100k-share order in a ~60k shares/min market this gives a total cost of
# roughly 10-30 bps, in line with institutional execution cost studies.
DEFAULT_ETA = 0.5
DEFAULT_GAMMA = 0.001


def default_impact_model() -> AlmgrenChrissImpactModel:
    """Calibrated Almgren-Chriss impact model shared by all environments and baselines."""
    return AlmgrenChrissImpactModel(eta=DEFAULT_ETA, gamma=DEFAULT_GAMMA, alpha=0.5)
