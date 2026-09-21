"""
Deep Reinforcement Learning Agents & Policy Routing wrappers.

Public API
----------
BaseAgentWrapper  – abstract interface all agents must implement
DQNAgent          – Stage 6: DQN without regime information
DQNTrainer        – MLflow-tracked training orchestrator
TrainingConfig    – fully serialisable training config dataclass
evaluate_agent    – roll out a trained agent and return AgentEvalResult
AgentEvalResult   – standardised eval result (mirrors BaselineResult)
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


# Lazy imports – avoids importing torch/SB3 unless actually needed
def __getattr__(name):
    if name == "DQNAgent":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent
    if name == "DQNTrainer":
        from src.agents.trainer import DQNTrainer
        return DQNTrainer
    if name == "TrainingConfig":
        from src.agents.trainer import TrainingConfig
        return TrainingConfig
    if name == "evaluate_agent":
        from src.agents.evaluator import evaluate_agent
        return evaluate_agent
    if name == "AgentEvalResult":
        from src.agents.evaluator import AgentEvalResult
        return AgentEvalResult
    raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
