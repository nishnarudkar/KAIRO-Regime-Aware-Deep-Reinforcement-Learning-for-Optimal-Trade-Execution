"""
tests/test_explainer.py — Unit Tests for Stage 12 Decision Explanation Layer

Tests:
  - Feature attribution calculation for standard (7-dim) state
  - Feature attribution calculation for regime-aware (12-dim) state
  - Percentage normalization (sum to 100%)
  - Regime influence score logic
  - Natural language summary generation
  - FastAPI explanation endpoints (/api/execution/explain & /api/execution/{id}/explain)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.agents.explainer import DecisionExplainer, STANDARD_FEATURE_NAMES, REGIME_FEATURE_NAMES
from src.api.main import app


@pytest.fixture
def explainer():
    return DecisionExplainer(delta=1e-3)


@pytest.fixture
def dummy_agent():
    class DummyQAgent:
        def get_q_values(self, state):
            state = np.asarray(state).flatten()
            res = np.ones(4, dtype=np.float32) * 0.25
            if len(state) >= 4:
                res[1] += state[0] * 0.5  # inventory impact on action 1
                res[2] += state[3] * 1.5  # volatility impact on action 2
            if len(state) >= 12:
                res[3] += state[11] * 2.0  # stress probability impact on action 3
            return res

        def predict(self, state):
            return int(np.argmax(self.get_q_values(state)))

    return DummyQAgent()


def test_explainer_standard_state(explainer, dummy_agent):
    state = np.array([1.0, 0.5, 0.001, 0.02, 0.0005, 1.0, 1.0], dtype=np.float32)
    result = explainer.explain_step(agent=dummy_agent, state=state, action=2)

    assert result["action"] == 2
    assert "action_label" in result
    assert len(result["feature_attributions"]) == 7
    assert len(result["feature_percentages"]) == 7
    assert result["regime_influence_score"] == 0.0
    assert sum(result["feature_percentages"].values()) == pytest.approx(100.0, abs=1e-3)
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_explainer_regime_aware_state(explainer, dummy_agent):
    state = np.array([1.0, 0.5, 0.001, 0.02, 0.0005, 1.0, 1.0, 3, 0.05, 0.05, 0.1, 0.8], dtype=np.float32)
    result = explainer.explain_step(agent=dummy_agent, state=state, action=3)

    assert result["action"] == 3
    assert len(result["feature_attributions"]) == 12
    assert len(result["feature_percentages"]) == 12
    assert result["regime_influence_score"] > 0.0
    assert sum(result["feature_percentages"].values()) == pytest.approx(100.0, abs=1e-3)


def test_api_explain_endpoint():
    client = TestClient(app)
    payload = {
        "state": [1.0, 0.5, 0.001, 0.04, 0.0005, 1.0, 1.0, 2, 0.05, 0.1, 0.75, 0.1],
        "action": 2,
        "policy": "Regime-Aware DQN",
    }
    response = client.post("/api/execution/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == 2
    assert "regime_influence_score" in data
    assert "feature_percentages" in data
    assert "summary" in data


def test_api_get_execution_explain():
    client = TestClient(app)
    # First simulate an execution run to populate store
    sim_res = client.post("/api/execution/simulate", json={
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 100000,
        "horizon_steps": 30,
        "policy": "Regime-Aware DQN",
        "scenario": "normal",
        "seed": 42
    })
    assert sim_res.status_code == 200
    exec_id = sim_res.json()["execution_id"]

    # Now explain the execution run
    exp_res = client.get(f"/api/execution/{exec_id}/explain")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert "action_label" in exp_data
    assert exp_data["regime_influence_score"] >= 0.0
