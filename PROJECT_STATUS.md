# Project Status

## Milestones

| Stage | Status | Notes |
|---|---|---|
| 0–5 Architecture, simulator, environment, data pipeline, HMM, baselines | Complete | VWAP was a copy of TWAP until Stage 16; now a real ex-ante volume-profile strategy |
| 6–7 DQN and regime-aware DQN | Complete | Trained on random windows (Stage 16); regime filter warm-started |
| 8–9 Experiment suite and PPO extension | Complete | Rebuilt on the unified protocol; 8 seeds × 6 scenarios × 30 paired windows |
| 10 FastAPI backend | Complete | Serves trained checkpoints; SQLite persistence; API-key-protected order routes |
| 11 Next.js UI | Complete | Redesigned; shows real model status, paired statistics, recorded results |
| 12 Decision explanations | Complete | Explains the trained network; labels match the environment |
| 13 Risk gates / paper trading | Complete (mock mode) | `/paper` and `/kill-switch` require `KAIRO_API_KEY` |
| 14 Docker | Complete | Python 3.12, non-root, build-time API URL, persistent volume |
| 15 Research audit | Rewritten | `docs/research_audit.md` reports measured results |
| **16 Validity overhaul** | **Complete** | Realistic data, random-window training, paired statistics, honest serving, CI |

## What the research shows

On the synthetic markets, regime-aware RL has **no robust advantage**: learned policies are about level with TWAP and
slightly behind POV, and the PPO gain from regime features is reproduced by a shuffled-regime control. See
[`docs/research_audit.md`](docs/research_audit.md).

## Leakage controls (all covered by tests)

| Risk | Control |
|---|---|
| Global feature scaling | Scalers fitted on the train split only |
| Lookahead volatility | Backward-looking windows only (`volatility` column is a causal 20-bar estimate) |
| HMM regime leakage | HMM fitted on train only; online forward filter; warm-start uses past bars only |
| Volume / price reference | Computed from bars before the episode window |
| VWAP profile | Estimated from history before the window |
| Train/test contamination | Chronological 70/30 split; training windows never touch the test region |

## Open items

* Validate on real market data (the Alpaca data client exists but the experiments are synthetic-only).
* Longer training / hyperparameter search; other horizons and order sizes.
* Per-regime analysis of where an agent could beat POV; better handling of unfillable orders.
* Real (non-mock) Alpaca paper execution behind the risk gates.
