"""
tests/test_api.py — Unit & Integration Tests for KAIRO FastAPI Server

Tests cover:
  - GET /health healthcheck endpoint
  - GET /api/baselines, /api/models, /api/experiments metadata endpoints
  - GET /api/regime/current market regime detection
  - POST /api/execution/simulate for baselines & RL policies
  - POST /api/execution/backtest multi-policy comparison
  - GET /api/execution/{id}, /metrics, and /trajectory
  - 404 and 400 error handling
"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.api.main import app
from src.api.store import global_store
from conftest import requires_models

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    """Clear execution store before each test."""
    global_store.clear()
    yield
    global_store.clear()


def test_health_check():
    """Verify GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_list_baselines():
    """Verify GET /api/baselines returns list of 3 baseline strategies."""
    response = client.get("/api/baselines")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    strategy_ids = {s["strategy_id"] for s in data}
    assert strategy_ids == {"TWAP", "VWAP", "POV"}


def test_list_models():
    """Verify GET /api/models returns list of 4 DRL models."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    model_ids = {m["model_id"] for m in data}
    assert model_ids == {"dqn_base", "dqn_regime", "ppo_base", "ppo_regime"}


def test_list_experiments():
    """Verify GET /api/experiments returns list of research experiment suites."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["status"] in ("completed", "not_run")


def test_get_current_regime():
    """Verify GET /api/regime/current detects market regime."""
    response = client.get("/api/regime/current?symbol=AAPL&scenario=normal&seed=42")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert 0 <= data["regime_id"] <= 3
    assert data["regime_label"] in {"Low Volatility", "Normal", "High Volatility", "Stress"}
    assert len(data["regime_probabilities"]) == 4


def test_simulate_execution_baseline():
    """Verify POST /api/execution/simulate runs TWAP simulation and stores record."""
    payload = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 50000.0,
        "horizon_steps": 20,
        "policy": "TWAP",
        "scenario": "normal",
        "seed": 42,
    }
    response = client.post("/api/execution/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "execution_id" in data
    exec_id = data["execution_id"]
    assert data["policy"] == "TWAP"
    assert data["target_inventory"] == 50000.0
    assert "metrics" in data
    assert data["metrics"]["completion_rate"] == 1.0

    # Query record by ID
    get_resp = client.get(f"/api/execution/{exec_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["execution_id"] == exec_id

    # Query metrics by ID
    metrics_resp = client.get(f"/api/execution/{exec_id}/metrics")
    assert metrics_resp.status_code == 200
    assert "implementation_shortfall_bps" in metrics_resp.json()

    # Query trajectory by ID
    traj_resp = client.get(f"/api/execution/{exec_id}/trajectory")
    assert traj_resp.status_code == 200
    assert len(traj_resp.json()["inventory_trajectory"]) > 0


@requires_models
def test_simulate_execution_rl():
    """Verify POST /api/execution/simulate runs DQN simulation."""
    payload = {
        "symbol": "MSFT",
        "side": "BUY",
        "quantity": 10000.0,
        "horizon_steps": 15,
        "policy": "DQN",
        "scenario": "normal",
        "seed": 42,
    }
    response = client.post("/api/execution/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["policy"] == "DQN"
    # 'completed' only when fully filled; otherwise honestly reported as 'partial'
    expected = "completed" if data["metrics"]["completion_rate"] >= 0.9999 else "partial"
    assert data["status"] == expected


def test_execution_not_found_404():
    """Verify 404 for non-existent execution ID."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/execution/{fake_id}").status_code == 404
    assert client.get(f"/api/execution/{fake_id}/metrics").status_code == 404
    assert client.get(f"/api/execution/{fake_id}/trajectory").status_code == 404


def test_simulate_invalid_scenario_400():
    """Verify 400 Bad Request for invalid scenario name."""
    payload = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 10000.0,
        "horizon_steps": 10,
        "policy": "TWAP",
        "scenario": "invalid_scenario_name",
    }
    response = client.post("/api/execution/simulate", json=payload)
    assert response.status_code == 400


def test_backtest_endpoint():
    """Verify POST /api/execution/backtest compares multiple policies."""
    payload = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 20000.0,
        "horizon_steps": 15,
        "policies": ["TWAP", "VWAP", "DQN"],
        "scenario": "normal",
        "seed": 42,
    }
    response = client.post("/api/execution/backtest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "backtest_id" in data
    assert len(data["results"]) >= 2
