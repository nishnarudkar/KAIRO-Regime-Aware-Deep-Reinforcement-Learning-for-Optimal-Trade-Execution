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

## 📊 API Endpoints

### 1. `POST /api/execution/explain`
Compute feature attribution for an arbitrary state vector and action.

**Request**:
```json
{
  "state": [1.0, 0.5, 0.001, 0.04, 0.0005, 1.0, 1.0, 2, 0.05, 0.1, 0.75, 0.1],
  "action": 2,
  "policy": "Regime-Aware DQN"
}
```

**Response**:
```json
{
  "action": 2,
  "action_label": "Accelerated (50% target rate)",
  "feature_attributions": {
    "remaining_inventory": 0.12,
    "time_remaining": -0.05,
    "volatility": 0.45,
    "prob_high_vol": 0.82
  },
  "feature_percentages": {
    "remaining_inventory": 14.5,
    "volatility": 25.2,
    "prob_high_vol": 42.8
  },
  "regime_influence_score": 52.4,
  "action_scores": {"0": -0.1, "1": 0.2, "2": 0.85, "3": 0.4},
  "action_advantages": {"0": -0.95, "1": -0.65, "2": 0.0, "3": -0.45},
  "summary": "Policy selected 'Accelerated (50% target rate)'. Key driving factors: prob_high_vol (42.8%) and volatility (25.2%). Market regime features contributed 52.4% to this decision."
}
```

### 2. `GET /api/execution/{id}/explain`
Retrieve explanation summary for a completed simulation run by UUID.

---

## 🔬 Unit Tests
Comprehensive unit tests in `tests/test_explainer.py` verify:
- Attribution score normalization ($\sum P_i = 100\%$)
- Finite-difference sensitivity computation
- Regime influence calculation for 7-dim vs 12-dim state vectors
- Natural language explanation generation
