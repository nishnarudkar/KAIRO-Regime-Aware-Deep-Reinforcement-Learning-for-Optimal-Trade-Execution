# Research Plan & Experimental Design

## Research Hypothesis
By incorporating real-time market regime classification (Low Volatility, Normal, High Volatility, Stress) into a Deep Reinforcement Learning (DRL) MDP framework, the trading policy will dynamically adjust execution urgency, mitigating market impact during high volatility and exploiting liquidity during low volatility, thereby outperforming traditional execution rules (TWAP, VWAP, POV) and baseline non-regime DRL agents in Implementation Shortfall (IS).

---

## Experimental Setup & Baseline Models

### Baseline Benchmark Strategies
1. **TWAP (Time-Weighted Average Price):**
   $$\nu_t = \frac{Q_0}{N}$$
   Executes equal order size across $N$ equal intervals.

2. **VWAP (Volume-Weighted Average Price):**
   $$\nu_t = Q_0 \cdot \frac{V_t}{\sum_{\tau=1}^N V_\tau}$$
   Executes order proportional to expected intra-day volume profile.

3. **POV (Percentage of Volume):**
   $$\nu_t = \min\left(q_t, \rho \cdot V_t^{\text{mkt}}\right)$$
   Executes fixed ratio $\rho \in (0.05, 0.20)$ of real-time market volume.

4. **Baseline DRL Agent (No Regime Input):**
   Standard PPO/DQN agent operating on price, volume, spread, inventory, and time remaining without regime features.

5. **Regime-Aware DRL Agent (Proposed):**
   PPO/DQN agent operating on the full MDP state vector including soft regime probabilities or discrete regime states.

---

## Performance Evaluation Metrics

### 1. Implementation Shortfall (IS)
$$IS = \frac{\text{Direction} \cdot (\bar{P}_{\text{exec}} - P_0)}{P_0} \times 10,000 \quad \text{(in basis points)}$$
where $P_0$ is the decision/arrival price, $\bar{P}_{\text{exec}}$ is the volume-weighted average fill price achieved by the execution agent, and $\text{Direction} \in \{+1 \text{ (Buy)}, -1 \text{ (Sell)}\}$.

### 2. Execution Slippage against VWAP Benchmark
$$\text{Slippage}_{\text{VWAP}} = \frac{\bar{P}_{\text{exec}} - P_{\text{VWAP}}}{P_{\text{VWAP}}} \times 10,000 \quad \text{(bps)}$$

### 3. Terminal Fill Rate
$$\text{Fill Rate} = 1.0 - \frac{q_T}{Q_0}$$
Target is 100% completion before deadline $T$.

### 4. Risk-Adjusted Cost (Gain-to-Pain / Sharpe of Execution Cost)
Evaluates execution cost variance across different market regimes to measure policy stability under stress.

---

## Planned Market Regimes
- **Regime 0 (Low Volatility):** Stable prices, high liquidity, tight bid-ask spreads.
- **Regime 1 (Normal):** Average historical volatility and volume profile.
- **Regime 2 (High Volatility):** Widening spreads, elevated price variance, clustering volume.
- **Regime 3 (Stress / Liquidity Crisis):** Extreme volatility spikes, severe order book asymmetry, liquidity gaps.

---

## Data Split & Validation Methodology
- **Train Period:** 60% chronologically ordered market data.
- **Validation Period:** 20% chronologically ordered market data (hyperparameter tuning).
- **Test Period:** 20% unseen, strictly out-of-time market data containing diverse market regimes.
- **K-Fold Walk-Forward Cross Validation:** Sliding 6-month train / 1-month test walk-forward splits to ensure stability across macro market shifts without lookahead leakage.
