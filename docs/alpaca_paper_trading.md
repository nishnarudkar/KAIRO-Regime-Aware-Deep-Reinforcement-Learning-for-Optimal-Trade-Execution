# Stage 13 — Alpaca Paper Trading & Pre-Trade Risk Gates Reference

## 🎯 Overview

Stage 13 introduces institutional-grade pre-trade safety controls and paper trading execution capabilities interfacing with the Alpaca Paper Trading API (`https://paper-api.alpaca.markets`).

---

## 🛡️ Pre-Trade Risk Gates (`ExecutionRiskGate`)

Before any order slice is routed to the paper broker or simulation environment, it must pass 4 deterministic safety gates:

| Risk Gate | Default Parameter | Description | Violation Action |
|---|---|---|---|
| **Max Notional Value** | `$1,000,000` | Maximum dollar value per single order slice ($P_t \times Q_t$) | Order Rejected |
| **Max Single Order Pct** | `50%` | Maximum slice shares allowed as % of total target inventory | Order Rejected |
| **Price Collar** | `3%` | Maximum price deviation relative to initial decision arrival price | Order Rejected |
| **Emergency Kill Switch** | `Inactive` | Global override flag halting all order routing instantly | Order Rejected |

---

## ⚡ API Endpoints

### 1. `POST /api/execution/paper`
Routes an order slice to Alpaca Paper Trading (or Mock Sandbox) after pre-trade risk gate validation.

**Request**:
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 10000.0,
  "current_price": 150.0,
  "arrival_price": 150.0,
  "target_inventory": 100000.0
}
```

**Successful Fill Response**:
```json
{
  "order_id": "paper-mock-a1b2c3d4",
  "status": "filled",
  "mode": "paper_mock",
  "symbol": "AAPL",
  "side": "BUY",
  "executed_quantity": 10000.0,
  "fill_price": 150.03,
  "reason": "Order passed risk gates and executed in Alpaca Paper Sandbox.",
  "timestamp": "2026-09-21T16:45:00Z"
}
```

**Rejection Response (e.g., Price Collar Breach)**:
```json
{
  "order_id": null,
  "status": "rejected",
  "mode": "paper_mock",
  "symbol": "AAPL",
  "side": "BUY",
  "executed_quantity": 0.0,
  "fill_price": 0.0,
  "reason": "Order rejected: Price collar breach (5.00% deviation vs arrival $150.00 exceeds limit 3.0%).",
  "timestamp": "2026-09-21T16:45:00Z"
}
```

### 2. `GET /api/execution/risk-status`
Returns active parameter limits and kill switch state.

### 3. `POST /api/execution/kill-switch`
Triggers or resets the emergency halt kill switch.

---

## 🔬 Unit Tests
Comprehensive tests in `tests/test_alpaca_risk_gates.py` verify:
- Max notional value rejection
- Max slice percentage rejection
- Price collar breach rejection
- Emergency kill switch activation and resetting
- Paper API mock fill execution
