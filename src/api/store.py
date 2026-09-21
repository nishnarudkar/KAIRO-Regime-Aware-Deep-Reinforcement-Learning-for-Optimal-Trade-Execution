"""
src/api/store.py — Thread-safe Execution Store for KAIRO API Server

Indexes simulated execution records, metrics, and trajectories in memory
by unique UUID string to serve /api/execution/{id}, /metrics, and /trajectory.
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional, Tuple, Any

from src.api.schemas import (
    ExecutionResponse,
    ExecutionMetricsResponse,
    ExecutionTrajectoryResponse,
)


class ExecutionStore:
    """In-memory thread-safe store for execution runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executions: Dict[str, ExecutionResponse] = {}
        self._trajectories: Dict[str, ExecutionTrajectoryResponse] = {}

    def save_execution(
        self,
        execution_resp: ExecutionResponse,
        trajectory_resp: ExecutionTrajectoryResponse,
    ) -> str:
        """
        Store an execution record and its trajectory.

        Returns:
            The execution_id (UUID string).
        """
        exec_id = execution_resp.execution_id
        with self._lock:
            self._executions[exec_id] = execution_resp
            self._trajectories[exec_id] = trajectory_resp
        return exec_id

    def get_execution(self, execution_id: str) -> Optional[ExecutionResponse]:
        """Fetch execution summary response by ID."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_metrics(self, execution_id: str) -> Optional[ExecutionMetricsResponse]:
        """Fetch metrics response by execution ID."""
        with self._lock:
            exec_record = self._executions.get(execution_id)
            if exec_record:
                return exec_record.metrics
            return None

    def get_trajectory(self, execution_id: str) -> Optional[ExecutionTrajectoryResponse]:
        """Fetch trajectory response by execution ID."""
        with self._lock:
            return self._trajectories.get(execution_id)

    def clear(self) -> None:
        """Clear all stored executions (useful for testing)."""
        with self._lock:
            self._executions.clear()
            self._trajectories.clear()


# Global singleton instance
global_store = ExecutionStore()
