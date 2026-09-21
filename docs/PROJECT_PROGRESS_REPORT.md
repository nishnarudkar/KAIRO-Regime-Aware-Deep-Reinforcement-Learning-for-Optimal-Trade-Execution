# Project Progress & Technical Overview

**Project:** KAIRO — Regime-Aware Deep Reinforcement Learning for Optimal Trade Execution

## Goal

Execute a large parent order (e.g. 100,000 shares) with minimal implementation shortfall, and test whether causally
inferred market regimes help a reinforcement-learning agent beat static baselines (TWAP, VWAP, POV).

## System

```
src/
├── evaluation/   regime-switching synthetic market, shared protocol, experiment runner, statistics
├── execution/    simulator with bid/ask, calibrated impact model, fees, risk gates
├── environment/  Gymnasium environments (7-dim plain, 12-dim regime-aware), reward in bps
├── regimes/      Gaussian HMM, causal forward filter, features
├── baselines/    TWAP, VWAP (ex-ante volume profile), POV
├── agents/       DQN / PPO wrappers, model registry, decision explainer
└── api/          FastAPI service (serves trained models, SQLite store, API-key routes)
frontend/         Next.js dashboard
scripts/          run_experiments, train_models, make_report
```

## Status

All planned stages are implemented. A Stage 16 audit found the original experiments invalid (5%-per-minute price
volatility, one replayed training path, a VWAP that was TWAP, on-the-fly 512-step "training" in the API, fabricated
explanations) and rebuilt the foundation; see [`research_audit.md`](research_audit.md) for the problems and fixes.

## Findings

With valid data and 8 seeds × 6 scenarios × 30 paired windows, regime-aware RL shows **no robust advantage**:
learned policies are level with TWAP, slightly behind POV, and the regime-feature gain seen for PPO is reproduced by a
shuffled-regime control. Results, statistics and limitations are in [`research_audit.md`](research_audit.md) and
`results/REPORT.md`.

## Next steps

Real-data validation, longer training and tuning, other horizons and order sizes, and non-mock paper execution.
