# Market Regime Detection Module Methodology & Architecture

## Executive Summary

Market regime detection aims to partition continuous financial market observations into discrete, latent macro-states (e.g., Low Volatility, Normal, High Volatility, Stress). Because true market regimes are unobservable, we employ a probabilistic **Gaussian Hidden Markov Model (HMM)**.

This document details the mathematical design, feature engineering, canonical state sorting, strict causal online inference protocol, and leakage prevention mechanisms.

---

## 1. Why Hidden Markov Models (HMM)?

Financial time series exhibit **regime-switching dynamics**, non-stationarity, volatility clustering, and heavy tails. Standard linear models assume constant variance and stationary dynamics, failing during regime shifts.

HMMs are ideal for market regime detection because:
1. **Latent State Architecture**: Models the market as a first-order Markov process where the true regime $S_t \in \{0, 1, 2, 3\}$ is unobserved, emitting observable market features $X_t$.
2. **State Transition Dynamics**: Captures state persistence and transition probabilities via a transition matrix $A_{ij} = P(S_t = j \mid S_{t-1} = i)$.
3. **Probabilistic Uncertainty**: Provides soft posterior belief distributions $P(S_t = k \mid X_{1:t})$ rather than hard deterministic labels.
4. **Online Causal Filtering**: Supports exact $O(K^2)$ real-time Bayesian updating (Forward algorithm) without lookahead leakage.

---

## 2. Selected Features (Causal & Microstructural)

All features are computed using **strictly past and present data at or before timestamp $t$**:

| Feature Name | Math / Formula | Financial Rationale |
|---|---|---|
| `log_return` | $r_t = \ln(P_t / P_{t-1})$ | Directional return drift and bar magnitude |
| `realized_vol_15m` | $\text{Std}(r_{t-14:t})$ | Backward rolling 15-minute return volatility |
| `parkinson_vol` | $\sqrt{\frac{(\ln(H_t/L_t))^2}{4 \ln 2}}$ | Intraday high-low volatility estimator |
| `relative_volume_15m` | $V_t / \text{MA}_{15}(V)$ | Volume spike / activity ratio vs 15-period mean |
| `hl_spread_proxy` | $(H_t - L_t) / C_t$ | Microstructure bid-ask spread & range proxy |

---

## 3. Number of Hidden States & Canonical Interpretation

We set $K = 4$ hidden states, representing distinct market micro-environments:

| State ID | Canonical Name | Expected Volatility | Interpretation & Characteristics |
|---|---|---|---|
| **0** | **Low Volatility** | Lowest ($\sigma \downarrow$) | Quiet, range-bound market with low spread and baseline volume |
| **1** | **Normal** | Moderate ($\sigma \sim \text{avg}$) | Standard continuous trading environment with balanced liquidity |
| **2** | **High Volatility** | Elevated ($\sigma \uparrow$) | Active trading, wider price swings, elevated volume |
| **3** | **Stress** | Highest ($\sigma \uparrow\uparrow$) | Extreme price shocks, illiquidity, wide spreads, rapid volatility spikes |

### Canonical State Sorting (Label Unidentifiability Fix)
Unsupervised HMM estimation is subject to **label switching** across training runs. To enforce strict interpretability:
- After EM fitting, hidden states are sorted in ascending order of emission volatility ($\mu_{\text{vol}}$).
- State 0 is strictly mapped to the lowest volatility state, and State 3 to the highest volatility/stress state.

---

## 4. Model Initialization & Parameter Fitting

- **Distributional Assumption**: Multivariate Gaussian emissions with full covariance matrices $\Sigma_k$.
- **Estimation Algorithm**: Expectation-Maximization (EM / Baum-Welch).
- **Initialization**: K-Means clustering initialization on training feature matrix.
- **Regularization**: Minimum covariance floor ($\epsilon = 10^{-3}$) added to diagonal of $\Sigma_k$ to prevent singular matrices during stress bars.

---

## 5. Leakage Prevention & Online Inference Protocol

### The Lookahead Problem in Naive HMM Implementations
In naive implementations, practitioners fit an HMM on the entire dataset $X_{1:T}$ and run Viterbi decoding or full-sample posterior smoothing. This uses future data $X_{t+1:T}$ to infer regime $S_t$ at time $t$, creating catastrophic lookahead leakage for RL training and trade execution.

### Strict Causal Filtering Solution
Our inference engine implements **HMM Forward Algorithm Filtering**:

$$P(S_t = j \mid X_{1:t}) = \frac{b_j(X_t) \sum_{i=0}^{K-1} P(S_{t-1} = i \mid X_{1:t-1}) A_{ij}}{\sum_{k=0}^{K-1} b_k(X_t) \sum_{i=0}^{K-1} P(S_{t-1} = i \mid X_{1:t-1}) A_{ik}}$$

Where:
- $b_j(X_t) = \mathcal{N}(X_t; \mu_j, \Sigma_j)$ is the emission likelihood of observation $X_t$ in state $j$.
- $A_{ij} = P(S_t = j \mid S_{t-1} = i)$ is the transition probability.

**Guarantees**:
1. At timestamp $t$, the inferred state $\hat{S}_t = \arg\max_k P(S_t = k \mid X_{1:t})$ depends **ONLY** on observations $X_1, \dots, X_t$.
2. Appending data points at $t+1, t+2, \dots$ does **NOT** retroactively modify past inferred regimes $\hat{S}_t$.
3. Scalers are fitted **ONLY** on the chronological training split (70%) and applied statelessly to validation/test splits.

---

## 6. Critical Disclaimer on Economic Truth

> **CRITICAL**: The inferred regime labels (0, 1, 2, 3) are **model-derived latent mathematical states** optimized under maximum likelihood of feature emissions. They do **not** represent external economic ground truth or fundamental news announcements. They serve as continuous/discrete statistical context indicators for trade execution algorithms.
