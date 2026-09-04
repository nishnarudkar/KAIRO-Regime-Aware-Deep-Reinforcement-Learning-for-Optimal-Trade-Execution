# Project Status & Pre-Implementation Risk Audit

## Milestone Progress

| Milestone | Status | Description |
|---|---|---|
| **Phase 1: Architecture & MDP Design** | **COMPLETED** | Repository layout, MDP definitions, research plan, configuration schemas, abstract module interfaces. |
| **Phase 2: Data Pipeline & Features** | PLANNED | Data loaders, feature engineering, order book metrics, lookahead prevention unit tests. |
| **Phase 3: Regime Detection Engine** | PLANNED | HMM & Volatility regime classifiers with rolling fit and temporal stability tests. |
| **Phase 4: Gymnasium Execution MDP** | PLANNED | Environment implementation, discrete action step dynamics, multi-term reward formulation. |
| **Phase 5: Baseline Strategies** | PLANNED | TWAP, VWAP, and POV execution strategies for benchmark comparison. |
| **Phase 6: DRL Agent Integration** | PLANNED | Stable-Baselines3 integration, regime feature routing, policy network tuning. |
| **Phase 7: Backtest & Evaluation** | PLANNED | Implementation Shortfall backtest suite, regime metric breakdowns, visual analytics dashboard. |
| **Phase 8: API Serving & Containerization**| PLANNED | FastAPI REST endpoints, MLflow tracking, Docker container deployment. |

---

## Assumptions Needing Empirical Validation
1. **Market Impact Model Parameters:** The temporary and permanent impact functions (Almgren-Chriss power-law constants) must be calibrated against real market tick data or order book liquidity.
2. **HMM State Stability:** Hidden Markov Model regime predictions must not flip erratically every tick; transition probabilities require smoothing or minimum regime hold durations.
3. **Discrete Action Granularity:** Discrete inventory fractions $\{0\%, 10\%, 25\%, 50\%\}$ must provide sufficient flexibility for optimal execution without causing excessive terminal inventory penalty.

---

## Potential Data Leakage Risks & Controls

| Leakage Risk | Mechanism | Mitigation / Prevention Strategy |
|---|---|---|
| **Global Feature Normalization** | Scaling features using global mean/std across train and test sets. | Fit scalers (StandardScaler / RobustScaler) STRICTLY on training split; apply `.transform()` on test/val windows. |
| **Lookahead Volatility Estimators** | Centered rolling windows or future return calculations. | Enforce backward-looking windows ONLY (`df['close'].pct_change().rolling(window, closed='left')`). |
| **HMM Regime Leakage** | Fitting HMM on the complete dataset including test periods. | Fit HMM on historical train split or maintain an online rolling window fit strictly using $\mathcal{F}_{t-1}$. |
| **Replay Buffer Pollution** | RL agent experiencing test-set transitions during training. | Complete physical separation of training market datasets and testing market datasets. |

---

## Unresolved Design Questions / Decisions

1. **Continuous vs Discrete Action Space:**
   - *Current Plan:* Discrete set $\{0\%, 10\%, 25\%, 50\%\}$ of remaining inventory.
   - *Consideration:* Should continuous execution fractions $[0, 1]$ be introduced in later iterations via SAC/PPO continuous heads?
2. **Order Book Granularity:**
   - *Decision Point:* Use 1-minute OHLCV with synthetic impact vs High-Frequency Level-2 Order Book tick replay.
3. **Regime Input Format to Policy:**
   - *Decision Point:* One-hot categorical regime vector vs Soft probability distribution $[P(R_0), P(R_1), P(R_2), P(R_3)]$ from HMM posterior.
