# Stage 11 — Next.js Product UI Documentation

## 1. Overview

Stage 11 introduces **KAIRO — Adaptive Execution Intelligence**, a production-quality **Next.js + TypeScript** web application providing an institutional quantitative trading dashboard connected to the FastAPI backend service (`http://127.0.0.1:8000`).

---

## 2. Main Screens & Features

### 1. New Execution (`NewExecutionTab.tsx`)
- Form to configure parent order parameters: Symbol (`AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOGL`, `AMZN`), Side (`BUY`/`SELL`), Target Quantity, Horizon Steps, Execution Policy (`TWAP`, `VWAP`, `POV`, `DQN`, `Regime-Aware DQN`, `PPO`, `Regime-Aware PPO`), Scenario (`normal`, `high_volatility`, `low_liquidity`, `stress`, `regime_transition`, `liquidity_shock`), and Seed.
- Triggers `POST /api/execution/simulate` and automatically transitions to Execution Monitor upon completion.

### 2. Execution Monitor (`ExecutionMonitorTab.tsx`)
- Real-time execution progress card with progress bar (0–100%).
- Current Price, Volatility, Spread, Remaining Inventory, Time Remaining, and Fill Rate.

### 3. Execution Analytics (`ExecutionAnalyticsTab.tsx`)
- Execution quality metric cards: Implementation Shortfall (IS bps & $), Market Impact Cost ($), Transaction Fees ($), Average Fill Price, Completion Rate (%), and VWAP Slippage (bps).
- Interactive Recharts trajectory plots:
  - **Inventory Decay Plot**: Remaining shares ($I_t$) over horizon steps.
  - **Market Price Trajectory Plot**: Asset price movements over time.
  - **Action Distribution Bar Chart**: Frequency of discrete inventory fill percentages (0%, 10%, 25%, 50%).

### 4. Strategy Comparison (`StrategyComparisonTab.tsx`)
- Runs multi-policy backtest via `POST /api/execution/backtest`.
- Winner highlight box identifying top-performing policy.
- Comparative Implementation Shortfall (IS bps) bar chart.
- Benchmark results data table comparing baselines and DRL agents.

### 5. Research & Regimes (`ResearchTab.tsx`)
- Online market regime detection card via `GET /api/regime/current`.
- Posterior probabilities distribution bar chart over the 4 canonical HMM regimes (Low Vol, Normal, High Vol, Stress).
- Research Questions (RQ1–RQ4) summary overview.

---

## 3. Running the Frontend Locally

```bash
# Navigate to frontend directory
cd frontend

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

Ensure the FastAPI backend server is running concurrently at `http://127.0.0.1:8000`:
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
