"""
src/api/store.py — Persistent execution store for the KAIRO API server

Execution records and trajectories are kept in SQLite so they survive restarts
and are shared by every worker process. The database path comes from
``KAIRO_DB_PATH`` (default ``data/kairo.db``; ``:memory:`` for an ephemeral store).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from src.api.schemas import (
    ExecutionMetricsResponse,
    ExecutionResponse,
    ExecutionTrajectoryResponse,
)

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "kairo.db"


class ExecutionStore:
    """SQLite-backed, thread-safe store for execution runs."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            path = self._db_path or os.environ.get("KAIRO_DB_PATH") or str(_DEFAULT_DB)
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS executions ("
                " id TEXT PRIMARY KEY,"
                " created_at TEXT NOT NULL,"
                " record TEXT NOT NULL,"
                " trajectory TEXT NOT NULL)"
            )
            self._conn.commit()
        return self._conn

    def save_execution(self, execution_resp: ExecutionResponse,
                       trajectory_resp: ExecutionTrajectoryResponse) -> str:
        with self._lock:
            conn = self._connection()
            conn.execute(
                "INSERT OR REPLACE INTO executions (id, created_at, record, trajectory) VALUES (?, ?, ?, ?)",
                (execution_resp.execution_id, execution_resp.timestamp,
                 execution_resp.model_dump_json(), trajectory_resp.model_dump_json()),
            )
            conn.commit()
        return execution_resp.execution_id

    def get_execution(self, execution_id: str) -> Optional[ExecutionResponse]:
        with self._lock:
            row = self._connection().execute(
                "SELECT record FROM executions WHERE id = ?", (execution_id,)).fetchone()
        return ExecutionResponse.model_validate_json(row[0]) if row else None

    def get_metrics(self, execution_id: str) -> Optional[ExecutionMetricsResponse]:
        record = self.get_execution(execution_id)
        return record.metrics if record else None

    def get_trajectory(self, execution_id: str) -> Optional[ExecutionTrajectoryResponse]:
        with self._lock:
            row = self._connection().execute(
                "SELECT trajectory FROM executions WHERE id = ?", (execution_id,)).fetchone()
        return ExecutionTrajectoryResponse.model_validate_json(row[0]) if row else None

    def clear(self) -> None:
        """Delete every stored execution (used by tests)."""
        with self._lock:
            conn = self._connection()
            conn.execute("DELETE FROM executions")
            conn.commit()


global_store = ExecutionStore()
