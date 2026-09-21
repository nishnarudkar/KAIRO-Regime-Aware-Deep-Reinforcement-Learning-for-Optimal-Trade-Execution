# PROJECT.md — Regime-Aware Deep RL for Optimal Trade Execution

This document is the single source of truth for this project. Read it in full before
doing any implementation work. It defines scope, sequencing, guardrails, and the
research standards this codebase must meet. Do not skip ahead in the sequence, do not
implement multiple stages in one pass, and do not relax any "hard rule" below without
explicitly flagging it to the user first.

---

## 1. Project Identity

- **Name:** Optimal Trade Execution with Regime-Aware Deep Reinforcement Learning
- **Repository:** `regime-aware-trade-execution`
- **Type:** Research-grade + product-grade system. Both halves matter — sloppy
  research invalidates the product, and a missing product layer means the research
  never gets used.

### Core Research Question
Can a regime-aware deep reinforcement learning agent reduce implementation shortfall
and execution cost compared with traditional execution strategies under changing
market conditions?

### Core Comparisons
1. TWAP
2. VWAP
3. POV
4. RL without regime information (DQN)
5. Regime-aware RL (Regime-Aware DQN, later PPO)

### Planned Market Regimes
- Low volatility
- Normal
- High volatility
- Stress

---

## 2. System Architecture (target end state)

```
Historical Market Data
        ↓
Feature Engineering
        ↓
HMM Regime Detection
        ↓
Custom Trade Execution MDP
        ↓
   ┌─────────┴─────────┐
   │                   │
Traditional         RL Agents
 Baselines          DQN / PPO
   │                   │
   └─────────┬─────────┘
        ↓
Evaluation Engine
        ↓
Research Results
        ↓
FastAPI Backend
        ↓
Next.js Product UI
        ↓
Optional Alpaca API
```

Target repository structure (proposed at Stage 0, may be refined but not
fundamentally changed without discussion):

```
src/
├── data/            # ingestion, cleaning, validation, storage
├── features/        # causal feature engineering
├── regimes/         # HMM regime detection
├── environment/      # Gymnasium MDP wrapping the simulator
├── execution/        # execution simulator, market impact, cost models
├── agents/           # DQN / PPO agent definitions
├── baselines/        # TWAP / VWAP / POV
├── evaluation/        # metrics, comparison, ablations
├── api/               # FastAPI service layer
└── utils/

tests/
docs/
config/
data/
```

---

## 3. Non-Negotiable Hard Rules

These apply to every stage, every prompt, every file. Violating any of these
invalidates the research and must never happen silently.

1. **No look-ahead / no data leakage.** At timestamp `t`, every feature, regime
   label, and decision must be computable using only information available at or
   before `t`. This includes returns, volume, volatility, spread, liquidity proxies,
   AND HMM regime labels. Never fit on the full dataset and back-apply labels as if
   they were known in real time.
2. **No fabricated results.** Never invent, simulate-and-hide, or hand-massage
   numbers to make a narrative work. If an experiment doesn't support a hypothesis,
   report that.
3. **No RL for appearance.** Every RL component must have a stated purpose tied to
   a research question. Don't add algorithms (e.g., SAC) without justification.
4. **No silent methodology changes.** If reward weights, splits, features, or model
   assumptions are simplified or changed for convenience, this must be explicitly
   surfaced to the user, not buried in a commit.
5. **Chronological splits only.** Never shuffle time-series data for train/val/test.
6. **Deterministic, seeded experiments.** Every run must record its seed, config,
   dataset version, and model version.
7. **Simulation before live money.** No real-money trading, ever. Alpaca integration
   (Stage 13) is paper-trading only, gated behind explicit risk checks.
8. **Secrets stay out of the repo.** All credentials via environment variables only.
9. **Explanation layer never controls decisions.** Any LLM-based explanation
   (Stage 12) is strictly post-hoc description of an already-chosen RL action — it
   never selects or overrides actions.
10. **One stage at a time.** Do not implement the next stage until the current one
    is reviewed and explicitly approved.

---

## 4. Development Sequence

Follow this order. Each stage = one prompt to Antigravity. Do not combine stages.
Do not proceed to stage N+1 until stage N's tests pass and have been reviewed.

| Week | Stage | Deliverable |
|------|-------|-------------|
| 1 | 0 | Project architecture only (no implementation) |
| 1 | 1 | Historical data pipeline |
| 2 | 2 | Execution simulator |
| 2 | 3 | Gymnasium MDP environment (no regime yet) |
| 3 | 4 | TWAP / VWAP / POV baselines |
| 4 | 5 | HMM regime detection (standalone) |
| 5 | 6 | First RL agent: DQN, no regime |
| 6 | 7 | Regime-Aware DQN + A/B experiment |
| 7–8 | 8 | Full research experiment suite + ablations |
| 9 | 9 | PPO extension |
| 10+ | 10 | FastAPI backend |
| 10+ | 11 | Next.js product UI |
| 10+ | 12 | Decision explanation layer |
| 10+ | 13 | Optional Alpaca integration (paper trading only) |
| 10+ | 14 | Dockerization |
| Final | 15 | Research & reproducibility audit |

---

## 5. Stage-by-Stage Prompts

Use these prompts verbatim (or lightly adapted) as separate Antigravity turns, in
order. After each one, run tests and require Antigravity to report back before
moving on (see Section 6).

### Stage 0 — Project Architect
```
You are the lead software architect for this project.

PROJECT: Optimal Trade Execution with Regime-Aware Deep Reinforcement Learning

GOAL: Build a research-grade and product-grade system for optimal execution of
large orders. Model trade execution as a Markov Decision Process and use Deep
Reinforcement Learning to learn an adaptive execution policy.

IMPORTANT:
Do NOT implement the entire project yet.
Do NOT create fake results.
Do NOT use future market information in any state or feature.
Do NOT add RL merely for appearance. Every RL component must have a clear purpose.

CORE RESEARCH QUESTION:
Can a regime-aware deep reinforcement learning agent reduce implementation
shortfall and execution cost compared with traditional execution strategies under
changing market conditions?

CORE COMPARISONS: TWAP, VWAP, POV, RL without regime information, Regime-aware RL

PLANNED MARKET REGIMES: Low volatility, Normal, High volatility, Stress

PLANNED MDP:
State: price/return features, volatility, trading volume, bid-ask spread,
liquidity-related features, remaining inventory, time remaining, market regime.
Actions: discrete execution quantities — 0%, 10%, 25%, 50% of remaining inventory.
Reward: penalize execution cost, implementation shortfall, market impact,
inventory risk, incomplete execution at the deadline.

The initial environment should use a historical market replay/simulation approach
rather than real-money trading.

TECHNOLOGY DIRECTION: Python, PyTorch, Gymnasium, Stable-Baselines3, scikit-learn,
hmmlearn, Pandas, NumPy, FastAPI, Next.js/React/TypeScript, PostgreSQL or Parquet,
MLflow, Docker.

Create ONLY the initial project architecture. Create: README.md, ARCHITECTURE.md,
RESEARCH_PLAN.md, PROJECT_STATUS.md, docs/ directory, src/ directory structure,
tests/ directory structure, config/ directory, data/ directory structure.

Propose a clean modular repository structure. For each module, explain its
responsibility. Do not write the RL implementation yet.

At the end, provide:
1. proposed folder structure
2. architecture explanation
3. MDP definition
4. development milestones
5. assumptions that require validation
6. potential data leakage risks
7. questions/decisions that must be resolved before implementation

Wait for further instructions after completing this architecture stage.
```

### Stage 1 — Data Pipeline
```
Implement ONLY the historical market-data pipeline.

Do not implement the RL agent yet. Do not implement the frontend. Do not
implement FastAPI. Do not fabricate market data unless explicitly creating a
clearly labeled synthetic-data test fixture.

DATA SOURCE: Use Alpaca's historical market-data API when credentials/
configuration are available.

The pipeline should support: downloading historical intraday market data, saving
raw data, cleaning data, sorting chronologically, validating timestamps, handling
missing values, removing duplicates, generating reproducible processed datasets.

INITIAL TARGET: Start with one liquid US equity such as AAPL. Use 1-minute bars.

Create src/data/: alpaca_client.py, downloader.py, cleaning.py, validation.py,
storage.py.

Create a feature pipeline that can later generate: returns, rolling volatility,
volume features, spread if quote data is available, liquidity proxies.

CRITICAL ANTI-LEAKAGE REQUIREMENT: At timestamp t, every feature must be computed
using only information available at or before t. Never use future prices, future
volume, future returns, future labels, or future regime information.

Add unit tests for: chronological ordering, duplicate removal, missing data
handling, timestamp alignment, look-ahead leakage detection.

Use environment variables for credentials. Never hardcode API keys.

Create a script/CLI that can download and process the data. Update README.md
with setup instructions.

Do not implement HMM or RL yet.
```

### Stage 2 — Execution Simulator
```
Now implement the trade execution simulation engine.

Do NOT implement DQN/PPO yet.

OBJECTIVE: Create a realistic but initially tractable market replay environment
for large-order execution.

INITIAL SCENARIO: Stock AAPL, side BUY, target order 100,000 shares, horizon 30
minutes, time step 1 minute.

The simulator must track: current market timestamp, current market price, bid/ask
when available, spread, volume, volatility, remaining inventory, executed
inventory, average execution price, elapsed time, remaining time, execution cost,
market impact, slippage.

Implement an ExecutionSimulator class supporting: market order execution,
configurable execution quantity, transaction costs, spread cost, market-impact
model, partial fills when appropriate, terminal incomplete-order penalty.

Use a clearly documented market-impact model. Keep the first model simple and
parameterized so it can later be replaced by a more realistic model.

IMPORTANT: Never use future data. Never use information after the current
timestep. Clearly separate observed market information from simulated outcomes.
All simulation assumptions must be documented in docs/execution_model.md.

Create unit tests for: inventory accounting, order completion, no negative
inventory, time advancement, execution cost, market impact, terminal conditions.

Also create a small deterministic test dataset so simulator tests are
reproducible.

At the end, create an example script showing: reset simulation → execute an
order → advance time → compute execution metrics → complete or terminate.

Do not implement RL yet.
```

### Stage 3 — MDP / Gymnasium Environment
```
Wrap the existing execution simulator in a Gymnasium-compatible environment.

Class: TradeExecutionEnv

FORMAL MDP:
State: [return, volatility, volume, spread, liquidity_proxy, remaining_inventory,
time_remaining, ...]. For this version DO NOT include market regime yet — we will
create a non-regime RL baseline first.

Action space (discrete):
0 = execute 0% of remaining inventory
1 = execute 10%
2 = execute 25%
3 = execute 50%
The selected percentage applies to the remaining inventory.

Reward: modular reward function with components for execution cost, market
impact, inventory risk, incomplete execution, transaction cost. Do NOT arbitrarily
choose weights without documenting them — put all reward coefficients in
configuration.

The environment should implement reset(), step(action), observation_space,
action_space, and an info dictionary exposing useful research metrics without
leaking future information.

Add: environment validation, random-action rollout, deterministic seeding, unit
tests, environment documentation.

Create docs/mdp.md documenting: state space, action space, transition mechanism,
reward function, termination condition, assumptions, Markov-property
considerations.

Use Gymnasium conventions correctly. Do not train any RL model yet.
```

### Stage 4 — Traditional Baselines
```
Implement the traditional execution baselines.

Create src/baselines/ with TWAP, VWAP, POV.

Each strategy must operate inside the same execution simulator/environment used
by the future RL agent.

TWAP: distribute execution approximately uniformly over the execution horizon.
VWAP: use historical/observed volume information available up to the current
execution context. Do not use future realized volume when making a decision.
POV: execute a configurable percentage of observed market volume.

All strategies must use the same initial order, execution horizon, market data,
transaction-cost model, and market-impact model.

Create a standardized evaluation interface so that all strategies return:
implementation shortfall, execution cost, market impact, average execution
price, completion rate, execution duration, turnover.

Create tests and a baseline comparison script.

IMPORTANT: Avoid look-ahead bias, especially in VWAP. Document exactly what
information is available at each timestep.

Do not implement RL yet.
```

### Stage 5 — HMM Regime Detection
```
Implement the market-regime detection module independently from the RL agent.

Use hmmlearn initially. Goal: detect latent market regimes from historical market
features. Potential regimes: Low volatility, Normal, High volatility, Stress.

Use causal/online-compatible features.

CRITICAL: The regime at timestamp t must be estimated using only data available
up to t. Avoid fitting an HMM using the full dataset and then pretending
historical regime labels were available in real time.

Create src/regimes/: hmm_model.py, features.py, inference.py, evaluation.py.

Support: training, model persistence, inference, chronological train/validation/
test splits, regime statistics.

Document: why HMM is being used, selected features, number of hidden states,
initialization, interpretation of states, leakage prevention.

Create plots/analysis scripts showing the detected regimes over historical data.

Do not connect HMM to RL yet.

At the end, generate a regime dataset where each timestamp has: timestamp +
available market features + causally inferred regime.

Do not claim that the regimes are economically "true"; they are model-derived
latent states.
```

### Stage 6 — First RL Agent (DQN, no regime)
```
Implement the first RL baseline using the existing TradeExecutionEnv.

Algorithm: DQN. Do not include HMM regime information yet.

Requirements: Stable-Baselines3, PyTorch, Gymnasium environment, deterministic
seeds, configurable hyperparameters, model checkpoints, training metrics,
evaluation script.

Training and evaluation must use chronological data splits. Do NOT randomly
shuffle time-series data.

Create src/agents/, src/training/, src/evaluation/.

The evaluation pipeline must calculate: implementation shortfall, execution
cost, market impact, completion rate, average execution price, inventory
trajectory, action distribution.

Compare DQN against TWAP, VWAP, POV. Do not claim the RL model is superior unless
the experiment demonstrates it.

Add MLflow experiment tracking if practical.

Create training config, evaluation config, reproducibility instructions. Save
all experiment parameters with each run.

Create scripts: python train.py / python evaluate.py.

Do not build the frontend yet.
```

### Stage 7 — Regime-Aware DQN
```
Extend the existing RL environment to support market-regime information.

Create a second environment/state configuration: RegimeAwareTradeExecutionEnv.

State should now contain: existing market features, order state, HMM-derived
market regime. The regime must be generated causally — the RL agent must not
receive future regime information.

Create two clearly separated configurations:
MODEL A: Standard DQN — State = Market + Order information
MODEL B: Regime-Aware DQN — State = Market + Order + HMM regime

Keep all other conditions identical: same data, same training period, same test
period, same action space, same reward, same simulation model.

Train and evaluate both. Create an experiment that directly measures whether
regime information improves execution quality.

Primary metrics: implementation shortfall, execution cost.
Secondary metrics: market impact, completion rate, turnover, inventory risk,
performance by regime.

Generate statistical summaries and plots. Do not assume Regime-Aware DQN will
outperform DQN — the experiment must objectively determine this.

Update docs/research_questions.md with:
RQ1: Does RL outperform conventional execution baselines?
RQ2: Does regime information improve RL execution?
RQ3: Does regime awareness improve robustness during regime transitions?
```

### Stage 8 — Research Experiment Suite
```
Build the research experiment framework.

Evaluate: TWAP, VWAP, POV, DQN, Regime-Aware DQN.

Design experiments for: normal market conditions, high volatility, low
liquidity, stress conditions, volatility regime transition, liquidity shock,
out-of-sample market period.

Use chronological train/validation/test separation.

Create an ablation study:
A. RL without regime
B. RL with regime
C. RL with shuffled/randomized regime labels as a control

The shuffled-regime control is important for determining whether improvements
come from useful regime information rather than merely adding another feature.

For every experiment record: configuration, random seed, model version, dataset
version, metrics, timestamp. Run multiple seeds where computationally feasible.

Generate: comparison tables, confidence intervals or variability estimates,
execution trajectory plots, inventory trajectories, implementation-shortfall
distributions, performance by regime.

Do not fabricate or manually alter results.

Create a reproducible experiment runner. Write results to a structured format
such as CSV/JSON/Parquet.

Create docs/experiments.md documenting the experimental methodology.
```

### Stage 9 — PPO Extension
```
Add PPO as a second RL algorithm. Keep the experimental setup identical to the
DQN experiments wherever possible.

Compare: DQN, PPO, Regime-Aware DQN, optionally Regime-Aware PPO if
computationally reasonable.

Do not redesign the project unnecessarily. The goal is to determine whether the
observed performance is specific to DQN or reproducible with another
policy-learning algorithm.

Use the same data split, environment, reward, baseline strategies, and
evaluation metrics.

Document why PPO is being evaluated and what research question it addresses.
Do not add SAC or other algorithms unless there is a clear research reason.
```

### Stage 10 — FastAPI Backend
```
Build a FastAPI backend around the validated research engine.

Do NOT modify the research logic unless required for clean integration.

Expose endpoints:
POST /api/execution/simulate
POST /api/execution/backtest
GET /api/execution/{id}
GET /api/execution/{id}/metrics
GET /api/execution/{id}/trajectory
GET /api/regime/current
GET /api/baselines
GET /api/models
GET /api/experiments

An execution request should accept: symbol, side, quantity, execution horizon,
selected policy, dataset/date range.

The backend should: validate input, load the requested model, obtain the
appropriate market data, detect the regime, run the execution environment,
return execution metrics and trajectory.

Do not allow unrestricted model retraining through public API endpoints. Keep
secrets out of the repository.

Add: Pydantic schemas, API error handling, logging, tests, OpenAPI
documentation. Separate research code from API/service code.
```

### Stage 11 — Next.js Product UI
```
Build a production-quality Next.js + TypeScript frontend for the Adaptive Trade
Execution platform. The product should NOT look like a generic stock-trading
dashboard. Position it as: "Adaptive Execution Intelligence".

Main screens:
1. New Execution — Symbol, BUY/SELL, Quantity, Execution horizon, Policy
   selection
2. Execution Monitor — current price, market regime, volatility, spread,
   remaining inventory, time remaining, latest RL action, execution progress
3. Execution Analytics — implementation shortfall, market impact, average fill
   price, completion rate, inventory trajectory, execution trajectory
4. Strategy Comparison — TWAP, VWAP, POV, DQN, Regime-Aware DQN
5. Research — performance by regime, ablation results, stress-test results,
   model comparison, experiment metadata

Use clean responsive UI and professional financial/research styling. Do not
create fake numbers — if an API result is missing, explicitly show that data is
unavailable.

Connect frontend to FastAPI backend through typed API clients.
```

### Stage 12 — Decision Explanation Layer
```
Add an explanation layer for RL execution decisions.

The explanation must be generated from structured model/environment
information. For each RL action, show: current market regime, remaining
inventory, time remaining, volatility, spread, liquidity, chosen execution
quantity, key reward components.

Do NOT allow an LLM to choose or override the RL action. The explanation layer
should only describe why the already-selected action is reasonable based on
observed inputs.

Make the explanation explicitly state that it is a post-hoc explanation of the
model decision, not a new trading recommendation.

Do not fabricate unavailable model reasoning or hidden chain-of-thought.
```

### Stage 13 — Optional Alpaca Integration
```
Add Alpaca integration as an OPTIONAL external-data and paper-trading layer.

IMPORTANT: The core research pipeline must remain fully reproducible without
live trading.

Support: historical market-data retrieval, optional paper-trading validation,
account/order status retrieval, order submission ONLY to a paper-trading
environment.

Create a strict separation:
RESEARCH MODE: Historical data → simulator → RL policy → metrics
PAPER MODE: Market data → trained policy → risk checks → paper order

Implement a risk gate before any paper order: maximum order size, maximum
notional exposure, maximum daily loss, kill switch, explicit paper-trading flag.

Never submit real-money orders. Keep all Alpaca credentials in environment
variables. Add tests using mocked API responses. Document exactly which parts
require an Alpaca account.
```

### Stage 14 — Production Packaging (Docker)
```
Containerize the complete application.

Services: frontend, backend, research/RL service if necessary, PostgreSQL,
MLflow if being used.

Create: Dockerfiles, docker-compose.yml, environment configuration, health
checks, startup instructions.

The system should be runnable locally with a small command sequence. Do not
place credentials in Dockerfiles or committed configuration files.

Update README.md with: architecture, local setup, training, evaluation, running
the application, running experiments, paper-trading setup if applicable.
```

### Stage 15 — Research & Reproducibility Audit
```
Perform a research and reproducibility audit of the entire project. Do NOT
change results.

Check for: look-ahead bias, data leakage, future regime information leakage,
incorrect train/test splitting, inconsistent baselines, reward-function
problems, incorrect inventory accounting, incorrect implementation-shortfall
calculations, unrealistic execution assumptions, RL environment violations,
non-deterministic experiments, missing random seeds, inconsistent evaluation
conditions, overfitting, missing ablation studies, claims unsupported by
experiments.

Review the project as if it were being prepared for a peer-reviewed research
paper.

Create docs/research_audit.md. For every issue: severity, affected file/module,
explanation, proposed fix.

Do not silently fix scientific methodology without documenting the change.
```

---

## 6. Standing Instructions for Every Stage

Append this to the end of every prompt sent to Antigravity:

> Run all relevant tests before declaring this stage complete. Then stop and
> report: (1) exactly what you implemented, (2) what assumptions you made,
> (3) what files changed, (4) what tests passed/failed. Do not proceed to the
> next milestone until I explicitly approve.

Watch specifically for Antigravity quietly "simplifying" methodology — e.g.
using future volume in VWAP because it's easier, changing reward weights without
updating config/docs, or shuffling time-series splits for convenience. Any such
shortcut must be flagged, not silently taken.

---

## 7. Definition of Done (per stage)

A stage is not complete until:
- [ ] All specified files/modules exist
- [ ] Unit tests exist and pass
- [ ] Relevant docs/*.md file is written and accurate
- [ ] No hard rule from Section 3 has been violated
- [ ] Antigravity has reported changes, assumptions, and test results
- [ ] The user has explicitly approved moving to the next stage