"""
src/evaluation/metrics.py — Metrics Aggregation and Statistical Summaries

Aggregates AgentEvalResult / BaselineResult objects across:
  - Multiple seeds
  - Multiple scenarios
  - Multiple strategies

Produces:
  - Per-strategy mean / std / min / max for every metric
  - Comparison tables (DataFrame)
  - Per-regime breakdown
  - Export to CSV / JSON / Parquet
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SingleRunRecord:
    """
    Flat record of one strategy evaluation (one seed, one scenario).
    Used to build the aggregate results DataFrame.
    """
    strategy_name: str
    scenario: str
    seed: int

    # Core execution metrics
    implementation_shortfall_bps: float
    execution_cost: float
    market_impact_cost: float
    total_transaction_fees: float
    completion_rate: float
    average_execution_price: float
    arrival_price: float
    vwap_slippage_bps: float
    terminal_penalty: float
    turnover: float

    # DRL-specific (NaN for baselines)
    reward_total: float = float("nan")
    action_0_pct: float = float("nan")   # % of steps choosing 0%
    action_1_pct: float = float("nan")   # % of steps choosing 10%
    action_2_pct: float = float("nan")   # % of steps choosing 25%
    action_3_pct: float = float("nan")   # % of steps choosing 50%

    @classmethod
    def from_baseline_result(
        cls,
        result,
        scenario: str,
        seed: int,
    ) -> "SingleRunRecord":
        return cls(
            strategy_name=result.strategy_name,
            scenario=scenario,
            seed=seed,
            implementation_shortfall_bps=result.implementation_shortfall_bps,
            execution_cost=result.execution_cost,
            market_impact_cost=result.market_impact_cost,
            total_transaction_fees=result.total_transaction_fees,
            completion_rate=result.completion_rate,
            average_execution_price=result.average_execution_price,
            arrival_price=result.arrival_price,
            vwap_slippage_bps=result.vwap_slippage_bps,
            terminal_penalty=result.terminal_penalty,
            turnover=result.turnover,
        )

    @classmethod
    def from_agent_eval_result(
        cls,
        result,
        scenario: str,
        seed: int,
    ) -> "SingleRunRecord":
        n_steps = len(result.action_trajectory) or 1
        counts = result.action_counts

        def _pct(idx: int) -> float:
            return 100.0 * counts.get(idx, 0) / n_steps

        return cls(
            strategy_name=result.agent_name,
            scenario=scenario,
            seed=seed,
            implementation_shortfall_bps=result.implementation_shortfall_bps,
            execution_cost=result.execution_cost,
            market_impact_cost=result.market_impact_cost,
            total_transaction_fees=result.total_transaction_fees,
            completion_rate=result.completion_rate,
            average_execution_price=result.average_execution_price,
            arrival_price=result.arrival_price,
            vwap_slippage_bps=result.vwap_slippage_bps,
            terminal_penalty=result.terminal_penalty,
            turnover=result.turnover,
            reward_total=result.reward_total,
            action_0_pct=_pct(0),
            action_1_pct=_pct(1),
            action_2_pct=_pct(2),
            action_3_pct=_pct(3),
        )


class ResultsAggregator:
    """
    Collects SingleRunRecords from multiple seeds and scenarios,
    then produces aggregated statistics and comparison tables.
    """

    def __init__(self) -> None:
        self._records: List[SingleRunRecord] = []

    def add(self, record: SingleRunRecord) -> None:
        self._records.append(record)

    def add_many(self, records: List[SingleRunRecord]) -> None:
        self._records.extend(records)

    def to_dataframe(self) -> pd.DataFrame:
        """Return raw records as a flat DataFrame."""
        if not self._records:
            return pd.DataFrame()
        rows = [asdict(r) for r in self._records]
        return pd.DataFrame(rows)

    def summary_table(
        self,
        metric: str = "implementation_shortfall_bps",
        scenarios: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregated mean ± std per strategy, optionally filtered by scenario list.

        Returns DataFrame with columns: strategy, mean, std, min, max, n_runs.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        if scenarios:
            df = df[df["scenario"].isin(scenarios)]

        agg = (
            df.groupby("strategy_name")[metric]
            .agg(mean="mean", std="std", min="min", max="max", n_runs="count")
            .reset_index()
            .rename(columns={"strategy_name": "strategy"})
            .sort_values("mean")
        )
        return agg

    def comparison_table(self, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Wide-format comparison: one row per strategy, columns = key metrics (mean).
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        if scenarios:
            df = df[df["scenario"].isin(scenarios)]

        metrics = [
            "implementation_shortfall_bps",
            "execution_cost",
            "market_impact_cost",
            "completion_rate",
            "vwap_slippage_bps",
            "terminal_penalty",
        ]
        return (
            df.groupby("strategy_name")[metrics]
            .mean()
            .reset_index()
            .rename(columns={"strategy_name": "strategy"})
            .sort_values("implementation_shortfall_bps")
        )

    def per_scenario_table(self, metric: str = "implementation_shortfall_bps") -> pd.DataFrame:
        """
        Pivot: rows = strategy, columns = scenario, values = mean(metric).
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        pivot = df.pivot_table(
            index="strategy_name",
            columns="scenario",
            values=metric,
            aggfunc="mean",
        ).reset_index()
        pivot.columns.name = None
        return pivot.rename(columns={"strategy_name": "strategy"})

    def rq_summary(self) -> Dict[str, float]:
        """
        Compute RQ1 and RQ2 improvement deltas (mean IS bps).
        Returns NaN for any missing strategy.
        """
        df = self.to_dataframe()
        means = df.groupby("strategy_name")["implementation_shortfall_bps"].mean()

        def _get(name: str) -> float:
            return float(means.get(name, float("nan")))

        best_baseline = float(np.nanmin([_get("TWAP"), _get("VWAP"), _get("POV")]))
        dqn    = _get("DQN")
        if np.isnan(dqn):
            dqn = _get("DQN (no regime)")   # legacy name
        ra_dqn = _get("Regime-Aware DQN")

        return {
            "best_baseline_is_bps": best_baseline,
            "dqn_is_bps": dqn,
            "regime_dqn_is_bps": ra_dqn,
            "rq1_dqn_vs_baseline_bps": best_baseline - dqn,
            "rq2_regime_vs_plain_dqn_bps": dqn - ra_dqn,
        }

    # ── Export ─────────────────────────────────────────────────────────────────

    def save_csv(self, path: str) -> None:
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Results saved → {path}")

    def save_parquet(self, path: str) -> None:
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Results saved → {path}")

    def save_json(self, path: str) -> None:
        rows = [asdict(r) for r in self._records]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info(f"Results saved → {path}")
