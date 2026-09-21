"""
tests/test_alpaca_risk_gates.py — Unit Tests for Stage 13 Risk Gates & Alpaca Paper Trading

Tests:
  - Risk gate order validation (all gates pass)
  - Max single order percentage breach rejection
  - Max notional value breach rejection
  - Price collar deviation breach rejection
  - Emergency kill switch activation & reset
  - AlpacaPaperExecutor mock slice execution & endpoint safety check
  - FastAPI paper trading & risk status endpoints
"""

import pytest
from fastapi.testclient import TestClient

from src.execution.risk_gates import ExecutionRiskGate, RiskGateConfig
from src.execution.alpaca_paper import AlpacaPaperExecutor
from src.api.main import app


@pytest.fixture
def risk_gate():
    config = RiskGateConfig(
        max_notional_value=100_000.0,
        max_single_order_pct=0.30,
        price_collar_pct=0.02,
        kill_switch_active=False,
    )
    return ExecutionRiskGate(config=config)


@pytest.fixture
def executor(risk_gate):
    return AlpacaPaperExecutor(risk_gate=risk_gate, mock_mode=True)


def test_risk_gate_passes_valid_order(risk_gate):
    passed, reason = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=20_000.0,
        current_price=4.0,
        arrival_price=4.0,
        target_inventory=100_000.0,
    )
    assert passed is True
    assert "passed" in reason.lower()


def test_risk_gate_max_order_pct_rejection(risk_gate):
    # 40k shares > 30% of 100k target inventory
    passed, reason = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=40_000.0,
        current_price=2.0,
        arrival_price=2.0,
        target_inventory=100_000.0,
    )
    assert passed is False
    assert "exceeds max single order limit" in reason


def test_risk_gate_max_notional_rejection(risk_gate):
    # 20k shares * $10 = $200k notional > $100k limit
    passed, reason = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=20_000.0,
        current_price=10.0,
        arrival_price=10.0,
        target_inventory=100_000.0,
    )
    assert passed is False
    assert "notional value" in reason.lower()


def test_risk_gate_price_collar_rejection(risk_gate):
    # Price $10.50 vs arrival $10.00 = 5% deviation > 2% collar limit
    passed, reason = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=5_000.0,
        current_price=10.50,
        arrival_price=10.00,
        target_inventory=100_000.0,
    )
    assert passed is False
    assert "price collar breach" in reason.lower()


def test_risk_gate_kill_switch_activation(risk_gate):
    risk_gate.set_kill_switch(True)
    passed, reason = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=1_000.0,
        current_price=10.0,
        arrival_price=10.0,
        target_inventory=100_000.0,
    )
    assert passed is False
    assert "emergency kill switch is active" in reason.lower()

    # Reset kill switch
    risk_gate.set_kill_switch(False)
    passed, _ = risk_gate.validate_order(
        symbol="AAPL",
        side="BUY",
        order_quantity=1_000.0,
        current_price=10.0,
        arrival_price=10.0,
        target_inventory=100_000.0,
    )
    assert passed is True


def test_executor_slice_execution(executor):
    res = executor.execute_slice(
        symbol="AAPL",
        side="BUY",
        quantity=5_000.0,
        current_price=10.0,
        arrival_price=10.0,
        target_inventory=100_000.0,
    )
    assert res["status"] == "filled"
    assert res["executed_quantity"] == 5_000.0
    assert res["fill_price"] > 0.0


def test_executor_safety_constraint():
    with pytest.raises(ValueError, match="Safety constraint violated"):
        AlpacaPaperExecutor(
            base_url="https://api.alpaca.markets",  # Live URL attempt without mock_mode
            api_key="real_key",
            secret_key="real_secret",
            mock_mode=False,
        )


def test_api_paper_and_risk_endpoints():
    client = TestClient(app)

    # Test GET /api/execution/risk-status
    r_resp = client.get("/api/execution/risk-status")
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert "max_notional_value" in r_data
    assert "kill_switch_active" in r_data

    # Test POST /api/execution/paper
    p_payload = {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 5000.0,
        "current_price": 100.0,
        "arrival_price": 100.0,
        "target_inventory": 100000.0,
    }
    p_resp = client.post("/api/execution/paper", json=p_payload)
    assert p_resp.status_code == 200
    p_data = p_resp.json()
    assert p_data["status"] == "filled"

    # Test POST /api/execution/kill-switch
    k_resp = client.post("/api/execution/kill-switch", json={"active": True})
    assert k_resp.status_code == 200
    assert k_resp.json()["kill_switch_active"] is True

    # Confirm order rejected when kill switch active
    p_resp2 = client.post("/api/execution/paper", json=p_payload)
    assert p_resp2.status_code == 200
    assert p_resp2.json()["status"] == "rejected"

    # Reset kill switch
    client.post("/api/execution/kill-switch", json={"active": False})
