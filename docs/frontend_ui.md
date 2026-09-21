# Next.js Product UI

**KAIRO — Adaptive Execution Intelligence** is a Next.js + TypeScript dashboard for the FastAPI backend
(`http://127.0.0.1:8000` by default; set `NEXT_PUBLIC_API_URL` at build time to change it).

Design: a quiet, dense trading-desk look. Graphite surfaces, hairline rules, IBM Plex Sans with tabular Plex Mono
figures, one accent colour; green / red / amber appear only where they carry meaning (buy/sell, fill, regime).

---

## Screens

### 1. Order Ticket (`NewExecutionTab.tsx`)
Parent-order form (symbol, side, quantity, horizon 5–120 min, seed), a policy table and a scenario picker, with a live
order summary rail. The policy table reads `/api/models`: learned policies without a trained checkpoint are disabled
and labelled instead of being run.

### 2. Monitor (`ExecutionMonitorTab.tsx`)
Fill progress, unfilled shares and terminal penalty, implementation shortfall, average fill price, slippage vs VWAP,
execution cost, and order / cost-component tables. Status is `completed` or `partial` as reported by the API.

### 3. Analytics (`ExecutionAnalyticsTab.tsx`)
Remaining-inventory and price trajectories, a per-step regime strip (regime-aware policies), and the action
distribution (learned policies only).

### 4. Comparison (`StrategyComparisonTab.tsx`)
Runs all policies over 1–30 identical out-of-sample windows. Shows mean shortfall with a 95% CI, the paired difference
vs TWAP with its CI (marked *n.s.* when the CI includes zero), cost and fill, and lists any policy that could not be
evaluated together with the reason.

### 5. Research (`ResearchTab.tsx`)
The live causal-HMM regime detector (posterior probabilities, with the true regime of the synthetic market for
reference) and the **recorded experiment results**: paired comparisons for RQ1–RQ3 with CI, p-value, seeds favouring
the treatment and a plain-language reading, selectable by scenario scope. Nothing on this screen is a fixed claim;
every verdict is computed from `results/`.

---

## Running

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000
npm run lint && npx tsc --noEmit && npm run build
```

Start the backend first (`uvicorn src.api.main:app --port 8000`). Screens that need trained models rely on
`python scripts/train_models.py` having been run.
