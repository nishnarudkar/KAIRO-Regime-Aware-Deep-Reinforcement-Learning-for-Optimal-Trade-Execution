"""
Deep Reinforcement Learning Agents & Policy Routing wrappers.
"""

from abc import ABC, abstractmethod

class BaseAgentWrapper(ABC):
    """Abstract interface for DRL agents."""

    @abstractmethod
    def train(self, total_timesteps: int) -> None:
        """Train DRL policy network."""
        pass

    @abstractmethod
    def predict(self, state: object, deterministic: bool = True) -> int:
        """Predict action index given state vector."""
        pass
