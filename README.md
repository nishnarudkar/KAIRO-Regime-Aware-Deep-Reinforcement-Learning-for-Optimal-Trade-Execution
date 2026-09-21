# KAIRO — Regime-Aware Deep Reinforcement Learning for Optimal Trade Execution

A research framework and demo platform for executing large parent orders with Markov Decision Processes, Hidden
Markov Model regime detection and deep reinforcement learning (DQN / PPO), evaluated against TWAP, VWAP and POV on a
regime-switching **synthetic** market.

> **Read this first.** On the synthetic markets used here, regime-aware RL does **not** show a robust advantage. Learned
> policies are about level with TWAP and slightly behind POV, and the apparent gains from regime information do not
> survive a shuffled-regime control. See [`docs/research_audit.md`](docs/research_audit.md) for the measured results
> and their limits. All data is synthetic; nothing here is evidence about live trading.

---

## Research questions

> **RQ1** Does deep RL outperform TWAP, VWAP and POV?
> **RQ2** Does causally inferred regime information improve RL execution?
> **RQ3** Does regime-awareness help in turbulent markets and beat a shuffled-regime control?
> **RQ4** Is the effect the same for value-based (DQN) and policy-gradient (PPO) agents?

| | Result (8 seeds × 6 scenarios × 30 paired windows, 100k training steps) |
|---|---|
| RQ1 | Not robust: regime-aware agents beat TWAP by ~1.6–1.8 bps of ~23; no learned policy beats POV |
| RQ2 | No clear evidence (DQN −0.25 bps, CI includes 0; PPO −3.6 bps, explained by the control) |
| RQ3 | Not supported: regime-aware PPO does not beat its shuffled control |
| RQ4 | No consistent effect across algorithms |

---

## Project status

**198 automated tests (100% passing), CI on every push.** Stages 0–16 complete (simulator, environment, HMM, baselines, agents, hyperparameter tuning, MLflow remote tracking, API, UI, explanations, paper-trading gates, Docker).

| Area | State |
|---|---|
| Synthetic market | Regime-switching, realistic per-minute scale, zero drift, bid/ask, causal volatility |
| Environment | Random-window episodes, causal observations, well-scaled state and reward, warm-started regime filter |
| Baselines | TWAP, VWAP (real ex-ante volume profile), POV |
| Agents | DQN, PPO, regime-aware variants and shuffled-regime controls; hyperparameter tuned & served from `models/` |
| MLflow Tracking | Remote experiment tracking integrated via DagsHub MLflow |
| Experiments | Paired multi-window evaluation, hyperparameter grid search, drift-free validation, real market data tests |
| API | Persistent (SQLite), API-key-protected order routes, restricted CORS, honest model status, errors surfaced |
| UI | Next.js 14 dashboard showing real model status, paired statistics and recorded experiment results |
| Packaging | Complete `requirements.txt`, Docker (Python 3.13, standalone Next.js container, non-root user), GitHub Actions CI |

---

## Screenshots

| View | Screenshot |
|---|---|
| **Order ticket** | ![Order ticket](screenshots/01_new_execution.png) |
| **Monitor** | ![Monitor](screenshots/02_execution_monitor.png) |
| **Analytics** | ![Analytics](screenshots/03_execution_analytics.png) |
| **Strategy comparison** (paired, with CIs) | ![Comparison](screenshots/04_strategy_comparison.png) |
| **Research** (regime detector + recorded results) | ![Research](screenshots/05_research_regimes.png) |

---

## Architecture

```mermaid
flowchart TD
    Gen[Regime-switching synthetic market] --> Feat[Causal regime features]
    Feat --> HMM[Causal HMM filter]
    Gen --> Env[Execution environment: random windows]
    HMM --> Env
    Env --> Agent[DQN / PPO, plain or regime-aware]
    Agent --> Sim[ExecutionSimulator: bid/ask, impact, fees]
    Sim --> Reward[Reward in bps]
    Reward --> Agent
    Sim --> Metrics[Implementation shortfall]
    Base[TWAP / VWAP / POV] --> Sim
    Metrics --> Stats[Paired windows, bootstrap CI, Wilcoxon]
    Agent --> Models[(models/ checkpoints)]
    Models --> API[FastAPI]
    Stats --> API
    API --> UI[Next.js dashboard]
```

---

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (source venv/bin/activate on macOS / Linux)
pip install -r requirements.txt  # Python 3.11+
```

### Run the tests

```bash
python -m pytest tests/                                          # 198 tests passing
```

### Train models & run hyperparameter tuning

```bash
python scripts/train_models.py                                   # writes models/*.zip + registry.json
python scripts/tune_agents.py                                    # hyperparameter tuning across learning rates & gamma
```

### DagsHub MLflow Remote Tracking

```bash
# Optional: Set credentials to stream experiment logs directly to DagsHub MLflow
$env:DAGSHUB_USERNAME="nishnarudkar"
$env:DAGSHUB_TOKEN="<your_token>"
python scripts/run_experiments.py --mlflow
```

### Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000            # docs at /docs
```

The repository already contains trained checkpoints and results, so you can start the API directly. Learned
policies without a checkpoint return `503` rather than a fabricated result.

### Run the dashboard

```bash
cd frontend
npm install
npm run dev                                                      # http://localhost:3000
```

### Reproduce the research results

```bash
python scripts/run_experiments.py --seeds 42 123 777 2024 31415 7 99 1234 --timesteps 100000 --jobs 16
python scripts/make_report.py                                    # writes results/REPORT.md from the result files
```

### Docker

```bash
docker compose up -d --build     # backend :8000, dashboard :3000
```

See [`docs/docker_deployment.md`](docs/docker_deployment.md) and `.env.example` for configuration.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/execution/simulate` | One policy on an out-of-sample window (`503` if the learned policy is untrained) |
| `POST` | `/api/execution/backtest` | Several policies over 1–30 identical windows; mean, CI, paired difference vs TWAP; `errors` for failures |
| `GET` | `/api/execution/{id}` `/metrics` `/trajectory` | Stored results (SQLite) |
| `GET` | `/api/execution/{id}/explain?step=N` | Explanation of the trained network's decision at a step |
| `POST` | `/api/execution/explain` | Explain a decision for a given state (trained policies only) |
| `POST` | `/api/execution/paper` 🔒 | Route a slice through the risk gates (mock paper executor) |
| `POST` | `/api/execution/kill-switch` 🔒 | Emergency halt |
| `GET` | `/api/execution/risk-status` | Risk-gate configuration |
| `GET` | `/api/regime/current` | Causal HMM regime at the latest bar |
| `GET` | `/api/models` | Trained status, training steps and held-out evaluation per model |
| `GET` | `/api/experiments` / `/api/experiments/results` | Experiment status and the recorded results |
| `GET` | `/api/baselines`, `/health` | Baselines, liveness |

🔒 requires `X-API-Key` matching `KAIRO_API_KEY`; refused when no key is configured. Full reference:
[`docs/fastapi_backend.md`](docs/fastapi_backend.md).

---

## Repository layout

```
config/            YAML documentation of parameters (checked against the code by a test)
docs/              methodology, audit, API / UI / Docker references
frontend/          Next.js 14 + TypeScript dashboard with standalone Docker configuration
models/            trained checkpoints, pooled HMM and registry.json (evaluation of each model)
results/           experiment outputs (archive_v1/, tuning/, real_data/)
scripts/           run_experiments.py, train_models.py, tune_agents.py, make_report.py
src/agents/        DQN / PPO wrappers, model registry, explainer
src/api/           FastAPI app, engine, security, SQLite store, routers
src/baselines/     TWAP, VWAP (ex-ante volume profile), POV, runner
src/environment/   TradeExecutionEnv, RegimeAwareTradeExecutionEnv, reward
src/evaluation/    scenarios, protocol (splits, windows, statistics), experiment runner, ablation control
src/execution/     ExecutionSimulator, impact models, risk gates, paper executor
src/regimes/       Gaussian HMM, causal forward filter, features
tests/             198 unit tests (causality, protocol, API hardening, end-to-end)
```

## Guardrails

1. **Causality.** Features, volatility, the volume reference, the VWAP profile and regime beliefs at step *t* use only
   bars up to *t*; tests change the future and assert nothing earlier moves.
2. **Chronological split.** Train on the first 70%, evaluate on the last 30%; the HMM is fitted on train only.
3. **Controls.** A shuffled-regime agent with the same input size isolates the value of regime information.
4. **Paired, replicated evaluation.** Every strategy runs on identical windows; results are replicated over seeds and
   reported with confidence intervals.
5. **Honest serving.** No training inside requests; untrained models are labelled and refused; failures are reported.

## Known limitations

Synthetic & real market evaluation; single order size and horizon; participation cap bounds completion rate in thin scenarios; Alpaca execution operates in paper/mock mode. Details in [`docs/research_audit.md`](docs/research_audit.md).
