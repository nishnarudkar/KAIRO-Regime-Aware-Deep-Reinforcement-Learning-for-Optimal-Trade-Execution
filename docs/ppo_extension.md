# Stage 9 — PPO Extension Methodology & Research Documentation

## 1. Overview

Stage 9 introduces **Proximal Policy Optimization (PPO)** as a second Deep Reinforcement Learning algorithm class alongside Deep Q-Networks (DQN).

### Research Purpose
The primary goal of evaluating PPO is to address **Research Question 4 (RQ4)**:
> **RQ4:** *Is the regime-aware performance improvement observed in Stage 7 specific to value-based RL architectures (DQN), or does it generalise to policy-gradient RL architectures (PPO)?*

If **Regime-Aware PPO (Model D)** outperforms **Standard PPO (Model C)** by a similar margin to how **Regime-Aware DQN (Model B)** outperforms **Standard DQN (Model A)**, we establish that market-regime features provide **algorithm-class-agnostic value** for trade execution.

---

## 2. Algorithmic Comparison: DQN vs. PPO

| Dimension | Model A / B (DQN) | Model C / D (PPO) |
|---|---|---|
| **RL Category** | Value-based, Off-policy | Policy-gradient, On-policy |
| **Optimization Target** | Action-Value Function $Q_\theta(s, a)$ | Direct Policy $\pi_\phi(a\|s)$ + Value Function $V_\psi(s)$ |
| **Update Mechanism** | Temporal Difference (TD) target fitting via Replay Buffer | Clipped Surrogate Objective over collected rollout batches |
| **Exploration Strategy** | $\epsilon$-greedy schedule ($\epsilon_{start}=1.0 \to \epsilon_{end}=0.05$) | Stochastic policy sampling ($\text{Softmax}$ over action logits) |
| **Network Architecture** | MLP `[128, 128]` | MLP `[128, 128]` (shared/policy network) |
| **Discount Factor ($\gamma$)** | `0.99` | `0.99` |

---

## 3. Experimental Setup & Controls

To guarantee strict scientific comparability, all 4 models are trained and evaluated under identical conditions:

1. **Market Data & Split**: Same synthetic/historical OHLCV data grid, chronologically split (70% train / 30% test). No time-series data shuffling.
2. **State Spaces**:
   - **Models A & C (No Regime)**: 7-dimensional observation vector $[I_t, T_{rem}, \Delta P_t, \sigma_t, S_{bid-ask}, V_t, \overline{V}]$
   - **Models B & D (Regime-Aware)**: 12-dimensional observation vector (7 base features + 1 categorical HMM regime ID + 4 posterior regime probabilities $P(S_t = k \| X_{1:t})$).
3. **Action Space**: Discrete action space $A \in \{0, 1, 2, 3\}$, mapping to target execution participation rates $[0\%, 50\%, 100\%, 200\%] \times \text{TWAP rate}$.
4. **Reward Function**: Identical implementation shortfall & inventory risk reward formulation:
   $$R_t = -\left[ (P_t - P_{arr}) \cdot v_t + \phi \cdot (\text{Spread}_t \cdot v_t) + \lambda \cdot I_t^2 \cdot \sigma_t^2 \right]$$
5. **Causality Enforcement**: HMM model is fitted **exclusively on the training split**. Forward filtering relies solely on past observations $X_{1:t}$.

---

## 4. Hyperparameter Choices (`config/ppo.yaml`)

```yaml
policy: "MlpPolicy"
n_steps: 512           # Rollout buffer length
batch_size: 64         # Mini-batch size for policy gradient updates
n_epochs: 10           # Optimization epochs per rollout
learning_rate: 0.0003  # Adam optimizer learning rate
gamma: 0.99            # Discount factor (identical to DQN)
gae_lambda: 0.95       # GAE advantage estimation lambda
clip_range: 0.2        # PPO clipping coefficient
policy_kwargs:
  net_arch: [128, 128] # Network architecture matching DQN
```

---

## 5. Decision Framework & Interpretation

Let:
- $\Delta_{DQN} = \text{IS}_{\text{Model A}} - \text{IS}_{\text{Model B}}$ (DQN Implementation Shortfall gain from regime, in bps)
- $\Delta_{PPO} = \text{IS}_{\text{Model C}} - \text{IS}_{\text{Model D}}$ (PPO Implementation Shortfall gain from regime, in bps)

### Outcomes & Research Inferences

1. **$\Delta_{DQN} > 0$ and $\Delta_{PPO} > 0$**:
   - **Conclusion**: Market regime information improves trade execution quality across both value-based and policy-based RL algorithms. High confidence in generalisable empirical benefit.
2. **$\Delta_{DQN} > 0$ and $\Delta_{PPO} \le 0$**:
   - **Conclusion**: Regime awareness benefit is specific to Q-value function estimation and does not directly transfer to PPO's actor-critic gradient updates.
3. **$\Delta_{DQN} \le 0$ and $\Delta_{PPO} > 0$**:
   - **Conclusion**: Policy gradient methods leverage smooth regime probability distributions better than discrete Q-learning.

---

## 6. Execution Instructions

Run the 4-model comparison CLI:
```bash
python scripts/run_ppo_experiment.py --timesteps 50000 --seed 42
```

Inspect MLflow runs:
```bash
mlflow ui
```
