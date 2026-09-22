# KAIRO — Presentation Script & Defense Guide

**Project:** KAIRO — Regime-Aware Deep Reinforcement Learning for Optimal Trade Execution  
**Target Duration:** 12–15 Minutes Presentation + 10 Minutes Q&A  
**Audience:** Academic Evaluators, Computer Science / Quantitative Finance Professors  

---

## Presentation Overview & Roadmap

| Section | Topic | Allocated Time |
| :--- | :--- | :--- |
| **Section 1** | Introduction & Problem Formulation (Implementation Shortfall & MDP) | 2.5 mins |
| **Section 2** | Regime Detection Architecture (Causal Gaussian HMM Filter) | 2.0 mins |
| **Section 3** | Experimental Design V2 & Two-Round Rigor Audit | 3.0 mins |
| **Section 4** | Empirical Findings Across 3 Evaluation Suites (Main, Long Horizon, Real AAPL) | 2.5 mins |
| **Section 5** | Full-Stack Platform, API, Safety Gates & MLflow Tracking | 2.0 mins |
| **Section 6** | Conclusion & Future Directions | 1.0 min |
| **Q&A** | Defensive Q&A Preparation (16+ Anticipated Professor Questions) | 10 mins |

---

## Slide-by-Slide Spoken Script

### Slide 1: Title & The Optimal Trade Execution Problem
*(Time: 0:00 - 1:15)*

**Visual:** Title Slide displaying *KAIRO Architecture Diagram* and *Parent Order Execution Problem Statement*.

**Spoken Script:**
> *"Good morning Professor and members of the evaluation committee. Today, I am excited to present **KAIRO: Regime-Aware Deep Reinforcement Learning for Optimal Trade Execution**.*
>
> *In institutional algorithmic trading, when a portfolio manager wants to buy or sell a large position—say, executing a parent order over a 30-minute to 90-minute horizon—executing that order all at once in a single market order triggers severe price impact, driving up purchasing costs. To prevent this, trading desks slice parent orders into smaller child orders over time.*
>
> *The fundamental metric we aim to minimize is **Implementation Shortfall (IS)**, defined as the basis-point difference between the decision price when the order was initiated and the volume-weighted average execution price achieved.*
>
> *Traditional static algorithms like TWAP (Time-Weighted Average Price), VWAP (Volume-Weighted Average Price), and POV (Percentage of Volume) execute child orders according to rigid schedule rules. The core objective of KAIRO is to determine whether adaptive Deep Reinforcement Learning agents—specifically Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO)—augmented with real-time, causally inferred market regime beliefs, can dynamically adjust execution schedules to outperform static baselines under strict risk constraints."*

---

### Slide 2: Mathematical MDP Formulation & Action-Reward Redesign
*(Time: 1:15 - 2:30)*

**Visual:** MDP Equations, State Vector Breakdown (7-dim plain vs 12-dim regime-aware), TWAP-Relative Action Space, and Current-Price Reward Control Variate.

**Spoken Script:**
> *"To address optimal execution with Reinforcement Learning, we formulate the execution task as a finite-horizon Markov Decision Process (MDP).*
>
> *Our state space preserves strict causality. Plain agents observe a 7-dimensional vector (remaining inventory, remaining time steps, returns, Parkinson volatility, bid-ask spread, order flow imbalance, participation rate). Regime-aware agents observe a 12-dimensional vector concatenating the posterior belief distribution over market regimes from a Hidden Markov Model, transition probabilities, and expected regime duration.*
>
> *A major innovation in our MDP formulation is our **TWAP-Relative Action Space**. In early RL implementations, action spaces used fixed fractions of remaining inventory (0/10/25/50%), which mathematically could not express TWAP—biasing the agent with a +1.7 bps penalty before training even began! In KAIRO, actions are multiples of the target TWAP slice (\( a \in \{0, 0.5, 1.0, 2.0, 4.0\} \times \text{Slice}_{\text{TWAP}} \)). Playing 1.0× exactly reproduces TWAP, allowing the agent to start at baseline parity and learn true execution alpha.*
>
> *Additionally, during training, we charge child-order fills against the *current* market price as a control variate, reducing episode-return variance from 87 bps std down to 3 bps std while preserving true arrival-price Implementation Shortfall for evaluation."*

---

### Slide 3: Causal Regime Detection & Gaussian HMM Filter
*(Time: 2:30 - 4:30)*

**Visual:** HMM Architecture Diagram, Feature Optimization (ARI 0.59–0.63), Forward Algorithm Equation, and Zero-Lookahead Warm-Start.

**Spoken Script:**
> *"Market conditions oscillate across structural regimes—such as low-volatility trending markets, high-volatility turbulence, or liquidity shocks. To detect these shifts without lookahead bias, KAIRO integrates an online **Gaussian Hidden Markov Model (HMM)**.*
>
> *The HMM models the unobserved regime state \( z_t \in \{1, \dots, K\} \) using observable market features. Through a tuning-seed feature search, we optimized the feature representation to reach an Adjusted Rand Index (ARI) of **0.59–0.63** against latent regimes, up from 0.42 in baseline models.*
>
> *To enforce zero-lookahead bias, HMM parameters are fitted exclusively on the 70% training split. During online simulation, we compute the posterior regime belief \( P(z_t = k \mid x_{1:t}) \) using the causal HMM Forward Algorithm:*
>
> \[
> \alpha_t(k) = P(x_t \mid z_t = k) \sum_{j=1}^K \alpha_{t-1}(j) A_{jk}
> \]
>
> *Furthermore, every test episode undergoes a 50-bar causal warm-start phase over preceding historical data so that the forward filter initializes with stable regime posterior probabilities rather than cold uniform priors."*

---

### Slide 4: Scientific Rigor & Two-Round Audit
*(Time: 4:30 - 7:30)*

**Visual:** Two-Round Audit Summary Table: Round 1 (Validity of Setup) & Round 2 (Eliminating Baseline Bias & Controls).

**Spoken Script:**
> *"Now, I would like to highlight a core pillar of this project: **Our Two-Round Research & Reproducibility Audit**.*
>
> *RL trading literature frequently suffers from hidden experimental flaws. We conducted a systematic two-round audit to establish a bulletproof benchmark:*
>
> * **Round 1 (Validity of Setup):**  
>   1. *Fixed synthetic price volatility from an unrealistic 5%/minute down to realistic scale (0.02%–0.3%/min) with zero drift.*  
>   2. *Replaced single-path training with random-window training across 5 seeds, 6 scenarios, and paired out-of-sample windows.*  
>   3. *Replaced fake TWAP-based 'VWAP' with a true ex-ante volume-profile VWAP.*  
>   4. *Incorporated scenario-specific bid-ask spreads and causal volume normalizers.*
>
> * **Round 2 (Eliminating Baseline Bias & Controls):**  
>   1. *Action space redesign to TWAP-relative multiples (0/0.5/1/2/4×).*  
>   2. *Dynamic order sizing set to 5.5% of trailing window volume (achieving ~100% fill rates everywhere).*  
>   3. *Training reward variance reduction via current-price control variates.*  
>   4. *Hyperparameter grid tuning across learning rates and network widths.*  
>   5. *Introduction of the **Shuffled-Regime Control**, which feeds permuted regime beliefs to isolate genuine regime signal processing from extra neural network parameter capacity."*

---

### Slide 5: Empirical Findings Across 3 Evaluation Suites (RQ1–RQ4)
*(Time: 7:30 - 10:00)*

**Visual:** Empirical Results Summary Table across Main Suite (30-min), Long Horizon (90-min), and Real AAPL Market Data Suite.

**Spoken Script:**
> *"We evaluated our models across three distinct experimental suites:*
>
> 1. **Main 30-Minute Suite (900 Paired Windows):**  
>    * All strategies operate within a narrow ~3 bps band (POV 20.08 bps, Regime PPO 21.90 bps, TWAP 22.59 bps, DQN 23.09 bps).  
>    * **RQ1 & RQ2:** Learned policies do not robustly beat TWAP (95% CIs straddle 0) and **no learned policy beats POV** (all +1.8 to +3.0 bps costlier).  
>    * **RQ3:** Regime-Aware PPO vs. Shuffled Control is **−0.08 bps (CI [−0.81, +0.66])**, proving that regime features provide no advantage over noise.
>
> 2. **Long Horizon 90-Minute Suite (480 Windows):**  
>    * **Critical Finding:** **DQN breaks down severely at 90 minutes** (+27–29 bps vs TWAP, IS = 54.70–60.99 bps, fill rate drops to 74.5%), revealing severe horizon instability when evaluated past its 30-min tuning window.  
>    * PPO remains stable (28.65 bps, level with TWAP at 28.67 bps).
>
> 3. **Real-Data Transfer Suite (5 Days Real AAPL 1-Min Bars, 63 Windows):**  
>    * Models trained purely on synthetic data transferred to real market bars without retraining.  
>    * Learned policies nominally beat TWAP (11.50 bps) and VWAP (11.10 bps)—Regime DQN achieved 10.14 bps and Regime PPO achieved 9.70 bps. However, Regime PPO fill rate dropped to 80%, highlighting real-world microstructural friction."*

---

### Slide 6: Production Engineering, Risk Gates & Full-Stack Platform
*(Time: 10:00 - 12:00)*

**Visual:** Full-Stack Architecture Diagram (FastAPI, Next.js 14 Dashboard, Paper Trading Risk Gates, DagsHub MLflow Tracking).

**Spoken Script:**
> *"Beyond empirical research, KAIRO is engineered as a complete production-grade execution platform:*
>
> * **Backend & API:** Built with FastAPI, serving trained model checkpoints from `models/` with SQLite persistence. It enforces honest serving—untrained models return HTTP 503 instead of mock data, and order routes are protected via API keys (`X-API-Key`).*
> * **Pre-Trade Risk Gates:** Our paper execution pipeline validates orders against institutional risk gates: Max Single Order Size, Max Daily Volume Cap, Fat-Finger Price Deviation limits, and an Emergency Kill-Switch.*
> * **Remote MLflow Tracking:** Integrated with DagsHub MLflow remote logging, capturing real-time loss curves, reward trajectories, hyperparameter configurations, and evaluation metrics.*
> * **Frontend Dashboard:** A Next.js 14 + TypeScript dashboard featuring 5 real-time views: Order Ticket, Execution Monitor, Performance Analytics with paired confidence intervals, Strategy Comparison, and HMM Regime Exploration.*
> * **Quality Assurance:** Covered by **198 unit tests** with 100% pass rate, validating causality, zero-lookahead, MDP boundaries, and API security."*

---

### Slide 7: Conclusion & Future Work
*(Time: 12:00 - 13:00)*

**Visual:** Key Takeaways Slide & Future Roadmap.

**Spoken Script:**
> *"In summary, KAIRO demonstrates that while deep reinforcement learning can match simple TWAP benchmarks, claims of massive DRL superiorities in literature often stem from flawed simulation setups, non-expressive action spaces, and missing ablation controls.*
>
> *Our rigorous multi-suite benchmark proves that standard regime-aware signals do not robustly beat POV baselines or shuffled controls under realistic market microstructures, and value-based algorithms like DQN can suffer severe horizon instability.*
>
> *Future research directions include extending training to multi-asset LOB (Limit Order Book) data, incorporating continuous action spaces with Soft Actor-Critic (SAC), and deploying non-mock live WebSocket paper trading via Alpaca Markets.*
>
> *Thank you. I am now open to your questions."*

---

## Defensive Q&A Guide (16+ Anticipated Professor Questions)

### Category 1: Action Space & Reward Engineering Mechanics

#### Q1: "Why did you switch from inventory-fraction actions to TWAP-relative actions in Round 2?"
* **What the Professor is testing:** Understanding of action-space design, baseline representation bias, and RL environment initialization.
* **Short Answer:** Fixed fraction actions (0/10/25/50% of remaining inventory) could not mathematically express TWAP, imposing an artificial +1.7 bps penalty on agents before training began.
* **Detailed Technical Answer:**
  > *"In Round 1, action choices were fractions of remaining inventory. Because remaining inventory decays exponentially over time under fractional execution, no sequence of discrete fraction choices could reproduce an equal-slice TWAP trajectory. In Round 2, we redefined actions as multiples of the target TWAP slice (\( a \in \{0, 0.5, 1.0, 2.0, 4.0\} \times \text{Slice}_{\text{TWAP}} \)). Action 1.0x plays exact TWAP. This ensures the agent starts at parity with TWAP and only learns true execution deviations."*

#### Q2: "Explain the 'current-price' reward control variate. How does it reduce training variance?"
* **What the Professor is testing:** Reinforcement learning variance reduction techniques, reward shaping, and control variates.
* **Short Answer:** Episode price drift over 30–90 minutes adds severe noise (~87 bps std) to episode returns. Charging fills against current market price during training isolates child-order execution cost (~3 bps std).
* **Detailed Technical Answer:**
  > *"When charging fills against initial arrival price \( P_0 \), random market trend over 30 bars creates massive reward noise that swamps gradient updates. By defining training step reward as:*
  >
  > \[
  > r_t^{\text{train}} = -\frac{P_{\text{exec}, t} - P_{\text{mid}, t}}{P_{\text{mid}, t}} \times 10,000 - \lambda \cdot \text{Spread Penalty}_t
  > \]
  >
  > *we remove un-forecastable market drift from the gradient signal. Evaluation always measures true arrival-price Implementation Shortfall (\( P_{\text{exec}} - P_0 \)), preserving standard benchmarking."*

---

### Category 2: Horizon Sensitivity & Long-Horizon Breakdown

#### Q3: "Why did DQN fail so severely (+27 to +29 bps IS) at the 90-minute horizon while PPO remained stable?"
* **What the Professor is testing:** Knowledge of value-based (Q-learning) vs. policy-gradient (PPO) generalization, horizon extrapolation, and Q-value overestimation.
* **Short Answer:** DQN was tuned for 30 steps. Extending to 90 steps causes Q-value overestimation compounding over longer episode horizons, whereas PPO's clipped policy bounds action probability shifts.
* **Detailed Technical Answer:**
  > *"DQN estimates state-action values \( Q(s, a) \) iteratively via temporal difference bootstrapping. When evaluated at 90 steps (3x its tuning horizon), accumulated Q-value approximation errors cause the policy to choose passive actions early, leaving large inventories at step 90 that trigger severe terminal liquidation penalties (+181 bps in `stress` scenario). PPO's stochastic policy with clipped advantage bounds parameter updates, preventing catastrophic policy degradation."*

#### Q4: "How did you fix the fill-rate drop in thin liquidity scenarios between Round 1 and Round 2?"
* **What the Professor is testing:** Parent order sizing relative to market volume, fill rate normalization, and market impact constraints.
* **Short Answer:** In Round 1, parent order size was fixed at 100,000 shares regardless of scenario volume. In Round 2, order size was dynamically scaled to 5.5% of trailing window volume.
* **Detailed Technical Answer:**
  > *"In Round 1, thin scenarios (`low_liquidity`, `liquidity_shock`) had less than 666,000 total shares available across 30 bars. Under a 15% participation cap, executing 100,000 shares was mathematically impossible, causing fill rates to drop to 22%–72%. In Round 2, parent order size \( Q_0 \) is set to \( 0.055 \times \bar{V}_{\text{window}} \), achieving ~100% fill rates across all scenarios and eliminating artificial liquidation penalty distortions."*

---

### Category 3: Hidden Markov Models & Regime Detection

#### Q5: "How did you improve the HMM Adjusted Rand Index (ARI) from 0.42 to 0.59–0.63 in Round 2?"
* **What the Professor is testing:** Feature engineering for unsupervised learning, clustering evaluation, and HMM state recovery.
* **Short Answer:** We conducted a grid search over feature combinations on tuning seeds, identifying Parkinson volatility, normalized spread, and log returns as the optimal emission feature set.
* **Detailed Technical Answer:**
  > *"In `src/regimes/hmm_detector.py`, the initial feature set included raw price momentum and volume ratios that introduced noise into Gaussian covariance matrices. By testing feature combinations against true latent generator regimes, we selected 1-minute log returns, rolling Parkinson volatility (\( \sigma_P \)), and normalized bid-ask spread. This increased regime classification ARI from 0.42 to 0.59–0.63 and accuracy to 55–56%."*

#### Q6: "How do you guarantee that the HMM regime belief contains zero lookahead bias during evaluation?"
* **What the Professor is testing:** Experimental integrity, data leakage prevention, and causal filtering.
* **Short Answer:** HMM parameters are fitted strictly on the 70% training split. Online regime posterior probabilities are updated using the HMM Forward Algorithm step-by-step using only observations up to step \( t \).
* **Detailed Technical Answer:**
  > *"We enforce zero lookahead bias in three ways: First, `fit()` is called only on `df_train`. Second, at test time, the forward variable \( \alpha_t(k) = P(x_t \mid z_t=k) \sum_j \alpha_{t-1}(j) A_{jk} \) processes observations sequentially up to bar \( t \). Third, we run a causal warm-start over 50 preceding historical bars before episode step 0 to stabilize posterior beliefs without touching future test data. Our unit test suite (`tests/test_regimes.py`) mutates future test bars and asserts that regime beliefs at step \( t \) remain bit-for-bit identical."*

---

### Category 4: Scientific Audit & Experimental Methodology

#### Q7: "Explain the Shuffled-Regime Control. Why is it necessary?"
* **What the Professor is testing:** Understanding of ablation studies, confounding variables, and baseline controls.
* **Short Answer:** Adding 5 regime features increases the network input dimension from 7 to 12. The shuffled control feeds permuted regime probabilities to prove whether gains stem from regime signals or extra network capacity.
* **Detailed Technical Answer:**
  > *"If a 12-dimensional Regime-Aware PPO agent outperforms a 7-dimensional Plain PPO agent, we cannot immediately attribute the gain to market regime intelligence—the improvement could simply be due to larger network capacity or multi-feature representation noise. In `src/evaluation/ablation.py`, the Shuffled-Regime Control receives identical 12-dim inputs, but the 5 regime belief features are randomly shuffled across time. Since Regime-Aware PPO (−0.08 bps vs Shuffled Control, CI [−0.81, +0.66]) failed to beat the shuffled control, we proved that the regime signal provided no true informational edge."*

#### Q8: "What did your Real-Data Transfer test reveal when evaluating synthetic-trained models on real AAPL bars?"
* **What the Professor is testing:** Sim-to-real transfer, real market microstructural friction, and model generalizability.
* **Short Answer:** Models nominally beat TWAP and VWAP by 0.5–1.8 bps on real AAPL data, but Regime PPO fill rate dropped from 99% to 80%, showing real-world microstructural friction.
* **Detailed Technical Answer:**
  > *"We evaluated synthetic-trained checkpoints directly on 5 trading days of real 1-minute AAPL bars (63 test windows) without retraining (`results/real_data/`). Regime DQN achieved 10.14 bps and Regime PPO achieved 9.70 bps vs TWAP (11.50 bps) and VWAP (11.10 bps). However, 3.2% of real minutes had zero trades, causing Regime PPO's fill rate to drop to 80.0% due to participation cap constraints. This demonstrates that while directional execution policies transfer reasonably well, microstructural illiquidity requires dedicated real-data fine-tuning."*

---

### Category 5: Statistical Rigor & Results Interpretation

#### Q9: "Why did you use Wilcoxon signed-rank tests instead of standard paired t-tests?"
* **What the Professor is testing:** Knowledge of parametric vs non-parametric statistics for financial returns.
* **Short Answer:** Implementation shortfall distributions across trading windows display fat tails (kurtosis) and skewness, violating the normality assumption of paired t-tests.
* **Detailed Technical Answer:**
  > *"Financial execution shortfall data across 900+ windows exhibits heavy tail distributions due to occasional liquidity shocks and terminal liquidation penalties. Standard paired t-tests are highly sensitive to outliers. The Wilcoxon signed-rank test is a non-parametric test that evaluates rank differences between paired strategy evaluations on identical windows, providing robust p-values without assuming Gaussian shortfall distributions."*

#### Q10: "If RL only beats TWAP by ~0.7 bps and loses to POV, is DRL actually useful for trade execution?"
* **What the Professor is testing:** Practical trading value, basis point economics, and realistic project evaluation.
* **Short Answer:** In institutional trading, 0.7 basis points on a $100M daily trading volume represents $7,000 per day ($1.75M/year) in savings. However, POV remains superior when volume profiles are predictable.
* **Detailed Technical Answer:**
  > *"While 0.7 bps sounds small, in institutional asset management executing billions annually, fractions of a basis point translate into substantial alpha retention. Furthermore, our findings show that static POV (which dynamic RL struggled to beat) benefits from knowing the volume structure. DRL's real promise lies in hybrid models—combining POV volume tracking with DRL dynamic urgency adjustments under adverse price momentum."*

---

### Category 6: Production Engineering & System Architecture

#### Q11: "How does your FastAPI backend prevent serving untrained or corrupted models to users?"
* **What the Professor is testing:** Software engineering safety, model registry design, and production readiness.
* **Short Answer:** The backend enforces a Model Registry (`models/registry.json`). Untrained or missing model checkpoints return an explicit HTTP 503 Service Unavailable error instead of fallback dummy data.
* **Detailed Technical Answer:**
  > *"In `src/agents/registry.py` and `src/api/routers/execution.py`, when a simulation request arrives for a specific agent (e.g., `ppo_regime`), the engine queries the model registry. If the checkpoint `.zip` file is missing or `is_trained` flag is false, the system raises `HTTPException(status_code=503, detail='Model ppo_regime is untrained')`. This guarantees honest API responses and prevents silent fallback failures."*

#### Q12: "How are pre-trade risk gates implemented in your system?"
* **What the Professor is testing:** Real-world trading risk compliance, fat-finger controls, and emergency protocols.
* **Short Answer:** Every paper trade order slice must pass through `AlpacaRiskGates` validating maximum slice size, fat-finger price deviation, daily volume cap, and emergency kill-switch status.
* **Detailed Technical Answer:**
  > *"In `src/execution/risk_gates.py`, before an order slice is routed to the paper execution pipeline, it undergoes four mandatory checks:
  > 1. **Max Slice Size Gate:** Asserts `slice_quantity <= max_slice_qty` (e.g., 5,000 shares).  
  > 2. **Fat-Finger Price Gate:** Asserts `|limit_price - mid_price| / mid_price <= 0.03` (3% max deviation).  
  > 3. **Daily Volume Cap:** Ensures cumulative executed shares do not exceed user-defined daily limits.  
  > 4. **Kill-Switch Check:** Rejects all outgoing orders instantly if the global kill-switch is triggered (`GET /api/execution/kill-switch`)."*

#### Q13: "How is experiment tracking managed across multiple training runs?"
* **What the Professor is testing:** MLOps, experiment reproducibility, and remote logging integration.
* **Short Answer:** We integrated DagsHub MLflow remote experiment tracking to log hyperparameters, reward curves, loss metrics, and artifact checkpoints directly to a cloud dashboard.
* **Detailed Technical Answer:**
  > *"In `src/utils/dagshub_utils.py`, the training and evaluation scripts initialize an MLflow tracking client connected to DagsHub (`https://dagshub.com/nishnarudkar/KAIRO-...mlflow`). During training runs, we log metrics at every 1,000 steps (episode reward, policy loss, value loss, implementation shortfall bps) along with full YAML run configurations. This provides complete visibility and remote auditing capability."*

---

### Key Formulas Cheat-Sheet for Blackboard / Slides

$$\text{Implementation Shortfall (bps)} = \frac{P_{\text{exec}} - P_0}{P_0} \times 10,000$$

$$\text{TWAP-Relative Action: } v_t = a_t \cdot \left(\frac{Q_0}{T}\right), \quad a_t \in \{0.0, 0.5, 1.0, 2.0, 4.0\}$$

$$\text{HMM Forward Belief Update: } \alpha_t(k) = P(x_t \mid z_t = k) \sum_{j=1}^K \alpha_{t-1}(j) A_{jk}$$

$$\text{Almgren-Chriss Price Impact: } P_{\text{exec}} = P_{\text{mid}} \pm \frac{\text{Spread}}{2} + \gamma \left(\frac{v_t}{V_t}\right) P_{\text{mid}} + \eta \cdot \text{sgn}(v_t) \sqrt{\frac{|v_t|}{\tau V_t}} \sigma_t$$

$$\text{Current-Price Training Reward: } r_t^{\text{train}} = -\frac{P_{\text{exec}, t} - P_{\text{mid}, t}}{P_{\text{mid}, t}} \times 10,000 - \lambda \cdot \text{Spread Penalty}_t$$
