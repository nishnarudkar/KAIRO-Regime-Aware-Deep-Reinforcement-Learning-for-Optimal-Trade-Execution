# Stage 10 — FastAPI Backend Service Documentation

## 1. Overview

Stage 10 wraps the validated KAIRO research engine (market simulator, HMM regime detector, baseline strategies, DQN, and PPO agents) in a high-performance **FastAPI backend service**.

### Key Architectural Principles
1. **Strict Separation of Concerns**: The API layer in `src/api/` imports and invokes the research engine modules without altering research contracts or reward calculations.
2. **Resource Protection**: Public API endpoints do **not** allow unrestricted model retraining or parameter grid searching.
3. **Execution Store & Persistence**: Execution simulation records, metrics, and step trajectories are indexed in a thread-safe in-memory store by UUID (`execution_id`).

---

## 2. API Endpoint Reference

### Health Check
- `GET /health`
  - **Response**: `{"status": "ok", "service": "KAIRO Adaptive Execution Intelligence API", "version": "1.0.0"}`

---

### Execution Engine Routes

#### 1. Simulate Single Execution
`POST /api/execution/simulate`

- **Request Body**:
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100000.0,
  "horizon_steps": 30,
  "policy": "Regime-Aware DQN",
  "scenario": "normal",
  "seed": 42
}
```

- **Response (200 OK)**:
```json
{
  "execution_id": "a4b2c1d3-8e9f-41a2-b3c4-d5e6f7a8b9c0",
  "timestamp": "2026-09-21T21:15:00+00:00",
  "symbol": "AAPL",
  "side": "BUY",
  "target_inventory": 100000.0,
  "executed_inventory": 100000.0,
  "remaining_inventory": 0.0,
  "policy": "Regime-Aware DQN",
  "scenario": "normal",
  "status": "completed",
  "metrics": {
    "implementation_shortfall": 11591.17,
    "implementation_shortfall_bps": 7.75,
    "execution_cost": 11591.17,
    "market_impact_cost": 450.20,
    "total_transaction_fees": 120.00,
    "terminal_penalty": 0.0,
    "completion_rate": 1.0,
    "average_execution_price": 150.11,
    "arrival_price": 150.00,
    "market_vwap_price": 150.86,
    "vwap_slippage_bps": -50.0,
    "action_counts": {"0": 5, "1": 10, "2": 10, "3": 5}
  }
}
```

#### 2. Multi-Policy Backtest Comparison
`POST /api/execution/backtest`

- **Request Body**:
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100000.0,
  "horizon_steps": 30,
  "policies": ["TWAP", "VWAP", "POV", "DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"],
  "scenario": "normal",
  "seed": 42
}
```

#### 3. Fetch Execution Record by ID
`GET /api/execution/{id}`

- Returns the `ExecutionResponse` object stored for the specified `execution_id`.
- Returns `404 Not Found` if the ID does not exist.

#### 4. Fetch Execution Metrics by ID
`GET /api/execution/{id}/metrics`

- Returns the `ExecutionMetricsResponse` object.

#### 5. Fetch Execution Trajectory by ID
`GET /api/execution/{id}/trajectory`

- **Response (200 OK)**:
```json
{
  "execution_id": "a4b2c1d3-8e9f-41a2-b3c4-d5e6f7a8b9c0",
  "inventory_trajectory": [100000.0, 95000.0, 85000.0, 70000.0],
  "action_trajectory": [1, 2, 2, 3],
  "price_trajectory": [150.00, 150.05, 150.10, 150.12]
}
```

---

### Market Regime Route

`GET /api/regime/current?symbol=AAPL&scenario=normal`

- **Response (200 OK)**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2026-09-21T21:15:00+00:00",
  "current_price": 150.45,
  "spread": 0.035,
  "volatility": 0.0021,
  "regime_id": 1,
  "regime_label": "Normal",
  "regime_probabilities": {
    "Low Volatility": 0.05,
    "Normal": 0.85,
    "High Volatility": 0.08,
    "Stress": 0.02
  }
}
```

---

### Metadata Routes

- `GET /api/baselines`: Returns list of available baseline execution strategies.
- `GET /api/models`: Returns metadata for trained DRL agents (DQN, Regime DQN, PPO, Regime PPO).
- `GET /api/experiments`: Returns list of research experiment suites and scenarios.

---

## 3. Running the Server Locally

Start the server using `uvicorn`:
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Interactive Documentation
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
