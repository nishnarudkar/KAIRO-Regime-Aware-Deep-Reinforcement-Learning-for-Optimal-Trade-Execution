# Formal MDP Specification for Optimal Trade Execution

## Overview

The `TradeExecutionEnv` formulates large-order execution as a discrete-time Markov Decision Process (MDP) in accordance with the standard Gymnasium interface framework.

The MDP models an agent executing a parent order of size $Q_0$ (e.g. 100,000 shares) over a discrete time horizon $T$ (e.g. 30 steps of 1-minute intervals).

---

## 1. State Space ($\mathcal{S}$)

The state observation vector $S_t \in \mathbb{R}^7$ at time step $t$ consists of exogenous market features and endogenous execution progress metrics:

$$S_t = \begin{bmatrix} r_t \\ \sigma_t \\ V_t^{\text{rel}} \\ S_t^{\text{rel}} \\ L_t \\ q_t^{\text{rel}} \\ \tau_t^{\text{rem}} \end{bmatrix}$$

| Dimension | Variable | Definition / Formula | Range | Description |
|---|---|---|---|---|
| 0 | Log Return ($r_t$) | $\ln(P_t / P_{t-1})$ | $(-\infty, \infty)$ | Instantaneous price return |
| 1 | Volatility ($\sigma_t$) | Rolling standard deviation of 1-min returns | $[0, \infty)$ | Instantaneous price volatility |
| 2 | Relative Volume ($V_t^{\text{rel}}$) | $V_t / \bar{V}$ | $[0, \infty)$ | Current bar volume relative to rolling mean volume |
| 3 | Relative Spread ($S_t^{\text{rel}}$) | $(P_t^{\text{ask}} - P_t^{\text{bid}}) / P_t^{\text{mid}}$ | $[0, \infty)$ | Relative bid-ask spread |
| 4 | Liquidity Proxy ($L_t$) | $V_t / (P_t^{\text{ask}} - P_t^{\text{bid}} + \epsilon)$ | $[0, \infty)$ | Volume to spread ratio (depth proxy) |
| 5 | Remaining Inventory Fraction ($q_t^{\text{rel}}$) | $q_t / Q_0$ | $[0, 1]$ | Fraction of target inventory unexecuted |
| 6 | Time Remaining Fraction ($\tau_t^{\text{rem}}$) | $(T - t) / T$ | $[0, 1]$ | Normalized remaining time horizon |

> **Note**: For this non-regime baseline phase, market regime categorical/probabilistic features are excluded.

---

## 2. Action Space ($\mathcal{A}$)

The action space is discrete with 4 choices representing fractions of the **remaining inventory** $q_t$:

$$\mathcal{A} = \{0, 1, 2, 3\}$$

| Action Index | Fraction ($\alpha_a$) | Executed Quantity Request ($u_t$) | Description |
|---|---|---|---|
| `0` | $0.00$ | $0$ shares | Hold / Do not execute in this bar |
| `1` | $0.10$ | $0.10 \times q_t$ | Execute 10% of remaining inventory |
| `2` | $0.25$ | $0.25 \times q_t$ | Execute 25% of remaining inventory |
| `3` | $0.50$ | $0.50 \times q_t$ | Execute 50% of remaining inventory |

Actual filled quantity $\tilde{u}_t$ is constrained by the remaining inventory and maximum bar volume participation cap $\rho_{\max} \cdot V_t$:

$$\tilde{u}_t = \min\left( u_t, q_t, \lfloor \rho_{\max} \cdot V_t \rfloor \right)$$

---

## 3. Transition Mechanism ($\mathcal{P}$)

State transitions are governed by the historical market data replay stream combined with endogenous inventory decay:

$$S_{t+1} = f(S_t, a_t, M_{t+1})$$

1. **Market Data Advance**: Exogenous market features $(P_{t+1}, V_{t+1}, \sigma_{t+1}, S_{t+1})$ advance to the next replay bar $t+1$.
2. **Inventory Decay**: $q_{t+1} = q_t - \tilde{u}_t$.
3. **Time Advance**: $t \leftarrow t + 1$, updating $\tau_{t+1}^{\text{rem}} = (T - (t+1)) / T$.
4. **Permanent Market Impact Propagation**: The executed quantity $\tilde{u}_t$ imparts a permanent shift $\Delta P_{\text{perm}}$ that elevates all future market prices $P_{t' > t}$.

---

## 4. Modular Reward Function ($\mathcal{R}$)

The reward at step $t$ is a negative penalty balancing execution cost, market impact, inventory risk, transaction fees, and terminal unexecuted inventory shortfall:

$$R_t = -\left( \lambda_{\text{cost}} \cdot c_t^{\text{shortfall}} + \lambda_{\text{impact}} \cdot c_t^{\text{impact}} + \lambda_{\text{risk}} \cdot c_t^{\text{risk}} + \lambda_{\text{fee}} \cdot c_t^{\text{fee}} + \lambda_{\text{terminal}} \cdot c_t^{\text{terminal}} \right)$$

### Components

1. **Execution Shortfall Cost ($c_t^{\text{shortfall}}$)**:
   $$c_t^{\text{shortfall}} = \frac{\tilde{u}_t \cdot (P_t^{\text{exec}} - P_0^{\text{mid}})}{Q_0 \cdot P_0^{\text{mid}}}$$

2. **Temporary Impact Cost ($c_t^{\text{impact}}$)**:
   $$c_t^{\text{impact}} = \frac{\tilde{u}_t \cdot |\Delta P_{\text{temp}, t}|}{Q_0 \cdot P_0^{\text{mid}}}$$

3. **Inventory Holding Risk ($c_t^{\text{risk}}$)**:
   $$c_t^{\text{risk}} = \left( \frac{q_t}{Q_0} \right)^2 \cdot \sigma_t^2$$

4. **Transaction Fee Cost ($c_t^{\text{fee}}$)**:
   $$c_t^{\text{fee}} = \frac{\text{Fee}_t}{Q_0 \cdot P_0^{\text{mid}}}$$

5. **Terminal Penalty ($c_T^{\text{terminal}}$)**:
   Applied strictly at $t = T$ if remaining inventory $q_T > 0$:
   $$c_T^{\text{terminal}} = \frac{q_T \cdot \left( P_T^{\text{ask}} + \phi_{\text{penalty}} \cdot \Delta P_{\text{temp}}(q_T, V_T, \sigma_T) - P_0^{\text{mid}} \right)}{Q_0 \cdot P_0^{\text{mid}}}$$

### Reward Hyperparameters (Configuration)
Default weights in `config/environment.yaml`:
- $\lambda_{\text{cost}} = 1.0$
- $\lambda_{\text{impact}} = 1.0$
- $\lambda_{\text{risk}} = 1.0 \times 10^{-4}$
- $\lambda_{\text{fee}} = 1.0$
- $\lambda_{\text{terminal}} = 10.0$

---

## 5. Termination Condition

An episode terminates (`terminated = True`) when either:
1. Remaining inventory is fully executed ($q_t = 0$).
2. The step index reaches the execution horizon ($t = T$).

Truncation (`truncated = True`) occurs if market data runs out before step $T$.

---

## 6. Markov Property & Assumptions

- **Markovian Assumption**: The state vector $S_t$ encapsulates localized market liquidity, volatility, and order progress metrics. While financial prices exhibit long-memory dynamics, including rolling return, volatility, relative volume, spread, and inventory metrics satisfies the approximate Markov property $P(S_{t+1} \mid S_t, A_t) \approx P(S_{t+1} \mid S_t, \dots, S_0, A_t \dots A_0)$.
- **No Future Data Leakage**: $S_t$ contains information generated strictly up to step $t$.
- **Causality**: Execution actions at step $t$ affect step $t$ execution price and future market prices $t+1, \dots, T$ via permanent market impact.
