# Stage 12 — Decision Explanation Layer & Interpretability Reference

## 🎯 Overview

The Decision Explanation Layer provides quantitative and qualitative interpretability for Deep Reinforcement Learning (DRL) trade execution policies. It decomposes policy actions into signed feature attributions, relative percentage contributions, and market regime influence scores.

---

## 🧮 Mathematical Formulation

### 1. Local Input Perturbation Attribution
For an observation vector $s \in \mathbb{R}^d$ and chosen action $a \in \{0, 1, 2, 3\}$, the feature sensitivity is computed via symmetric finite difference on the policy Q-values $Q(s, a)$ or action probabilities $P(a \mid s)$:

$$\frac{\partial Q(s, a)}{\partial s_i} \approx \frac{Q(s + \delta e_i, a) - Q(s - \delta e_i, a)}{2\delta}$$

where $\delta = 10^{-3}$ and $e_i$ is the $i$-th basis vector.

The feature attribution $A_i$ scales local sensitivity by feature magnitude:

$$A_i = \frac{\partial Q(s, a)}{\partial s_i} \cdot (|s_i| + \epsilon)$$

### 2. Relative Percentage Contribution (%)
The relative contribution $P_i$ of feature $i$ is the normalized absolute magnitude:

$$P_i = \frac{|A_i|}{\sum_{j=1}^d |A_j|} \times 100\%$$

### 3. Market Regime Influence Score (%)
For 12-dimensional regime-aware environments, the **Regime Influence Score** aggregates percentage contributions across regime state features (regime ID $S_t^{\text{regime}}$ and posterior probabilities $P(S_t=k)$):

$$\text{RegimeInfluenceScore} = \sum_{k \in \text{RegimeFeatures}} P_k$$

In standard non-regime environments (7-dimensional state), $\text{RegimeInfluenceScore} = 0\%$.

---

## Feature and action definitions

The explainer's feature names and action meanings match the environments exactly.

| Index | Feature | Unit |
|---|---|---|
| 0 | `log_return` | percent |
| 1 | `volatility` | percent (causal 20-bar realized) |
| 2 | `volume_ratio` | bar volume / mean of the preceding history |
| 3 | `spread_bps` | quoted spread in bps |
| 4 | `liquidity` | ln(1 + volume / spread) − 13 |
| 5 | `remaining_inventory` | fraction of the parent order |
| 6 | `time_remaining` | fraction of the horizon |
| 7 | `regime_id` | 0–3 (regime-aware policies only) |
| 8–11 | `prob_low_vol`, `prob_normal`, `prob_high_vol`, `prob_stress` | causal HMM posterior |

Actions: `0` wait, `1` execute 10% of remaining inventory, `2` 25%, `3` 50%.

## What is explained

Explanations are computed on the **trained policy network** loaded from disk (DQN Q-network or PPO policy logits).
They are local, finite-difference sensitivities of that network: they describe what the network is responsive to at
one state, not causal effects in the market. Baselines have no network and cannot be explained (`400`).

## API

### `POST /api/execution/explain`

```json
{ "policy": "Regime-Aware DQN", "state": [0.02, 0.05, 1.1, 1.4, 0.3, 0.8, 0.5, 1, 0.1, 0.7, 0.15, 0.05], "action": 2 }
```

`action` is optional; when omitted the policy's own greedy action is explained. The state must have the policy's
dimension (7, or 12 for regime-aware) or the call returns `400`. A policy without a trained checkpoint returns `503`.

### `GET /api/execution/{id}/explain?step=N`

Explains the decision the policy made at step *N* of a stored execution, using the observation it actually saw
(recorded in the execution's trajectory).

The response contains `action`, `action_label`, `feature_attributions` (signed), `feature_percentages`,
`regime_influence_score` (share of attribution on regime features; 0 for non-regime policies), `action_scores`
(Q-values or logits), `action_advantages` and a text `summary`.
