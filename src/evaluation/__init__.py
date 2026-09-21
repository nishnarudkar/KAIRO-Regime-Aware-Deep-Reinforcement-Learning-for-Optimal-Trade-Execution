"""
Evaluation, metrics calculation, and backtesting suite.

Public API
----------
BaseEvaluator        — abstract interface (legacy)
ResultsAggregator    — collects SingleRunRecords, produces summary tables
SingleRunRecord      — flat record of one strategy × seed × scenario
generate_scenario_data — generate synthetic market data for a named scenario
SCENARIOS            — dict of available scenario configs
run_experiment_suite — full Stage 8 experiment orchestrator
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseEvaluator(ABC):
    """Abstract interface for backtesting and evaluating execution strategies."""

    @abstractmethod
    def evaluate(self, strategy_results: pd.DataFrame) -> dict:
        """Calculate Implementation Shortfall, VWAP slippage, and fill rates."""
        pass


def __getattr__(name):
    if name in ("ResultsAggregator", "SingleRunRecord"):
        from src.evaluation.metrics import ResultsAggregator, SingleRunRecord
        return locals()[name]
    if name == "generate_scenario_data":
        from src.evaluation.scenarios import generate_scenario_data
        return generate_scenario_data
    if name == "SCENARIOS":
        from src.evaluation.scenarios import SCENARIOS
        return SCENARIOS
    if name == "run_experiment_suite":
        from src.evaluation.experiment_runner import run_experiment_suite
        return run_experiment_suite
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
