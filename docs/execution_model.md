# Trade Execution Simulation Model

## Overview

The `ExecutionSimulator` provides a high-fidelity, deterministic market replay engine for simulating the execution of large parent orders (e.g., 100,000 shares of AAPL over a 30-minute horizon with 1-minute timesteps).

The design strictly enforces **temporal causality**:
- At time step $t$, only historical and current market information up to step $t$ ($\mathcal{F}_t$) is accessible.
- Future market data ($\mathcal{F}_{t' > t}$) is completely inaccessible to prevent lookahead bias.
- Simulated outcomes (executed shares, remaining inventory, price impact, costs) are strictly segregated from true exogenous market observations.

---

## 1. Initial Scenario Configuration

- **Asset**: AAPL (or any liquid asset with tick/1-min bar data)
- **Side**: `BUY` (or `SELL`)
- **Target Quantity ($Q_0$)**: 100,000 shares
- **Execution Horizon ($T$)**: 30 steps (e.g., 30 minutes with $\Delta t = 1$ minute)
- **Time Step ($\Delta t$)**: 1 minute

---

## 2. Mathematical Model & Execution Formulas

### 2.1 State Variables

At timestep $t \in [0, T-1]$:

| Variable | Symbol | Description |
|---|---|---|
| Market Timestamp | $\tau_t$ | Current market timestamp at step $t$ |
| Mid Price | $P_t^{\text{mid}}$ | Unaffected baseline market mid-price |
| Bid / Ask | $P_t^{\text{bid}}, P_t^{\text{ask}}$ | Current top of book quotes |
| Spread | $S_t$ | Bid-Ask spread $P_t^{\text{ask}} - P_t^{\text{bid}}$ |
| Volume | $V_t$ | Market bar volume in timestep $t$ |
| Volatility | $\sigma_t$ | Instantaneous asset volatility |
| Remaining Inventory | $q_t$ | Unexecuted parent order inventory ($q_0 = Q_0$) |
| Executed Inventory | $E_t$ | Total shares executed so far ($E_0 = 0, E_t = \sum_{k=0}^{t-1} u_k$) |
| Order Request | $u_t$ | Requested execution quantity at step $t$ ($0 \le u_t \le q_t$) |

---

### 2.2 Partial Fills & Participation Cap

To prevent unrealistic execution of massive volume in low-liquidity bars, execution is subject to a maximum participation rate cap $\rho_{\max} \in (0, 1]$ (default: 15% of market volume $V_t$):

$$\tilde{u}_t = \min\left(u_t, q_t, \lfloor \rho_{\max} \cdot V_t \rfloor\right)$$

where $\tilde{u}_t$ is the actual filled quantity for step $t$.

---

### 2.3 Market Impact Model

We employ a generalized **Almgren-Chriss** style market impact model decomposing price impact into **Temporary Impact** and **Permanent Impact**.

#### Temporary Impact ($\Delta P_{\text{temp}}$)
Temporary impact affects only the current order execution price due to instantaneous liquidity consumption:

$$\Delta P_{\text{temp}}(u_t, V_t, \sigma_t) = \text{side} \cdot \eta \cdot \left( \frac{\tilde{u}_t}{V_t} \right)^\alpha \cdot P_t^{\text{mid}}$$

where:
- $\text{side} = +1$ for `BUY`, $-1$ for `SELL`
- $\eta$ is the temporary impact coefficient (e.g., $0.1$)
- $\alpha$ is the volume power law exponent (default: $0.5$ for square-root law or $1.0$ for linear model)

#### Permanent Impact ($\Delta P_{\text{perm}}$)
Permanent impact permanently shifts the underlying market mid-price forward into future timesteps due to information leakage:

$$\Delta P_{\text{perm}}(\tilde{u}_t, V_t) = \text{side} \cdot \gamma \cdot \left( \frac{\tilde{u}_t}{V_t} \right) \cdot P_t^{\text{mid}}$$

where $\gamma$ is the permanent impact parameter.

The accumulated permanent impact modifies future market mid-prices:

$$P_{t+1}^{\text{mid, affected}} = P_{t+1}^{\text{mid, unobserved}} + \sum_{k=0}^{t} \Delta P_{\text{perm}}(\tilde{u}_k, V_k)$$

---

### 2.4 Spread Cost & Transaction Fees

#### Spread Cost
Executing via market orders incurs cross-the-spread costs:
- For `BUY`: Execution starts at the Ask price ($P_t^{\text{ask}} = P_t^{\text{mid}} + \frac{1}{2} S_t$)
- For `SELL`: Execution starts at the Bid price ($P_t^{\text{bid}} = P_t^{\text{mid}} - \frac{1}{2} S_t$)

#### Transaction Fees & Commissions
Commissions consist of a per-share fee $c_{\text{share}}$ and per-trade fixed fee $c_{\text{trade}}$, or basis point fee $c_{\text{bps}}$:

$$\text{Fee}_t = c_{\text{trade}} + c_{\text{share}} \cdot \tilde{u}_t + c_{\text{bps}} \cdot \left( \tilde{u}_t \cdot P_t^{\text{exec, base}} \right)$$

---

### 2.5 Total Effective Execution Price

For a `BUY` order:

$$P_t^{\text{exec}} = P_t^{\text{ask}} + \Delta P_{\text{temp}}(\tilde{u}_t, V_t, \sigma_t) + \frac{\text{Fee}_t}{\tilde{u}_t} \quad (\text{if } \tilde{u}_t > 0)$$

Average execution price across all steps:

$$\bar{P}_t = \frac{\sum_{k=0}^{t} \tilde{u}_k \cdot P_k^{\text{exec}}}{\sum_{k=0}^{t} \tilde{u}_k}$$

---

### 2.6 Execution Metrics & Performance Accounting

#### 1. Implementation Shortfall (IS)
Implementation Shortfall measures execution cost against the initial arrival mid-price benchmark $P_0^{\text{mid}}$:

$$\text{IS} = \text{side} \cdot \sum_{t=0}^{T-1} \tilde{u}_t \left( P_t^{\text{exec}} - P_0^{\text{mid}} \right)$$

In basis points:

$$\text{IS}_{\text{bps}} = \frac{\text{IS}}{Q_0 \cdot P_0^{\text{mid}}} \times 10,000$$

#### 2. Volume Weighted Average Price (VWAP) Benchmark
Market VWAP over the execution window:

$$\text{VWAP}_{\text{market}} = \frac{\sum_{t=0}^{T-1} P_t^{\text{mid}} \cdot V_t}{\sum_{t=0}^{T-1} V_t}$$

Execution relative to VWAP:

$$\text{Slippage}_{\text{VWAP}} = \text{side} \cdot (\bar{P}_{T-1} - \text{VWAP}_{\text{market}})$$

#### 3. Terminal Incomplete Order Penalty
If remaining inventory $q_T > 0$ at horizon end $t = T$:
Remaining shares are forcibly liquidated at the final price plus a penalizing market impact multiplier $\phi_{\text{penalty}}$:

$$\text{Penalty}_{\text{terminal}} = q_T \cdot \left( P_T^{\text{ask}} + \phi_{\text{penalty}} \cdot \Delta P_{\text{temp}}(q_T, V_T, \sigma_T) \right)$$

This strongly disincentivizes leaving unexecuted inventory at the end of the horizon.

---

## 3. Simulator State & Interface API

The `ExecutionSimulator` exposes clean separation between observed market snapshot and simulator execution state:

```python
state = simulator.get_state()
# Snapshot dict contains:
# - timestamp, current_price, bid, ask, spread, volume, volatility
# - remaining_inventory, executed_inventory, average_execution_price
# - elapsed_steps, remaining_steps, is_done
# - total_execution_cost, total_impact_cost, total_slippage_bps
```
