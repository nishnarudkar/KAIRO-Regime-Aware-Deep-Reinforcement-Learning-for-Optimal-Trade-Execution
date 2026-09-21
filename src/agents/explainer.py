"""
src/agents/explainer.py — Decision Explanation Layer for DRL Execution Policies

Provides post-hoc explainability for RL and baseline execution decisions:
  - Feature contribution / attribution scores via local input perturbation
  - Relative feature magnitude percentages (%)
  - Regime influence score (% attribution from regime features)
  - Q-value / action score counterfactual advantage
  - Natural language decision summary generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# Standard and Regime State Feature Names
# Must match TradeExecutionEnv / RegimeAwareTradeExecutionEnv observation layouts.
STANDARD_FEATURE_NAMES = [
    "log_return",
    "volatility",
    "volume_ratio",
    "spread_bps",
    "liquidity",
    "remaining_inventory",
    "time_remaining",
]

REGIME_FEATURE_NAMES = STANDARD_FEATURE_NAMES + [
    "regime_id",
    "prob_low_vol",
    "prob_normal",
    "prob_high_vol",
    "prob_stress",
]


# Action meanings must match the environment. "fraction" mode: [0, 10, 25, 50]% of remaining inventory;
# "twap_multiple" mode (the default in experiments and served models): multiples of the TWAP slice.
ACTION_MEANINGS_FRACTION = {
    0: "Wait (execute 0% of remaining inventory)",
    1: "Execute 10% of remaining inventory",
    2: "Execute 25% of remaining inventory",
    3: "Execute 50% of remaining inventory",
}
ACTION_MEANINGS_TWAP = {
    0: "Pause (0x the TWAP slice)",
    1: "Slow (0.5x the TWAP slice)",
    2: "On schedule (1x the TWAP slice)",
    3: "Accelerate (2x the TWAP slice)",
    4: "Rush (4x the TWAP slice)",
}
ACTION_MEANINGS = ACTION_MEANINGS_FRACTION   # backwards-compatible name


class DecisionExplainer:
    """
    Post-hoc decision explainer for trade execution agents.

    Supports:
      - Stable-Baselines3 DQN / PPO models
      - BaseAgentWrapper instances
      - Arbitrary baseline policy rules
    """

    def __init__(self, delta: float = 1e-3):
        """
        Args:
            delta: Perturbation step size for finite-difference feature sensitivity.
        """
        self.delta = delta

    def explain_step(
        self,
        agent: Any,
        state: np.ndarray,
        action: int,
        feature_names: Optional[List[str]] = None,
        action_meanings: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute post-hoc feature attributions and regime influence for a single execution step.

        Args:
            agent: DRL agent (wrapper or SB3 model) or callable predicting Q-values/logits.
            state: Observation vector array of shape (7,) or (12,).
            action: Action index chosen by the policy.
            feature_names: Optional custom feature name list matching state dimension.
            action_meanings: Optional dictionary mapping action indices to text descriptions.

        Returns:
            Dictionary containing:
              - action: Chosen action integer index
              - action_label: Human-readable action description
              - feature_attributions: Signed feature attribution values
              - feature_percentages: Absolute magnitude attribution percentages (%)
              - regime_influence_score: Total % of decision weight from regime features
              - action_scores: Predicted Q-values or probabilities for all actions
              - action_advantages: Score gap relative to chosen action
              - summary: Human-readable natural language explanation
        """
        state = np.asarray(state, dtype=np.float32).flatten()
        dim = len(state)

        if feature_names is None:
            if dim == 12:
                feature_names = REGIME_FEATURE_NAMES
            elif dim == 7:
                feature_names = STANDARD_FEATURE_NAMES
            else:
                feature_names = [f"feature_{i}" for i in range(dim)]

        # Step 1: Compute Q-values or action scores across all actions
        action_scores = self._evaluate_action_scores(agent, state)
        nb_actions = len(action_scores)
        if action_meanings is None:
            action_meanings = ACTION_MEANINGS_TWAP if nb_actions == 5 else ACTION_MEANINGS_FRACTION
        chosen_score = float(action_scores[action]) if action < nb_actions else 0.0

        # Step 2: Compute feature sensitivities via finite difference on chosen action score
        sensitivities = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += self.delta
            state_minus[i] -= self.delta

            scores_plus = self._evaluate_action_scores(agent, state_plus)
            scores_minus = self._evaluate_action_scores(agent, state_minus)

            sens_plus = scores_plus[action] if action < len(scores_plus) else 0.0
            sens_minus = scores_minus[action] if action < len(scores_minus) else 0.0

            sensitivities[i] = (sens_plus - sens_minus) / (2.0 * self.delta)

        # Signed attribution = sensitivity * feature value (or magnitude deviation)
        attributions = sensitivities * (np.abs(state) + 1e-4)

        abs_attribs = np.abs(attributions)
        total_abs = np.sum(abs_attribs)
        if total_abs > 1e-8:
            percentages = (abs_attribs / total_abs) * 100.0
        else:
            percentages = np.ones(dim, dtype=np.float32) * (100.0 / dim)

        # Step 3: Compute Regime Influence Score (%)
        regime_indices = [
            idx for idx, name in enumerate(feature_names) if "regime" in name or "prob_" in name
        ]
        if regime_indices:
            regime_pct = float(np.sum(percentages[regime_indices]))
        else:
            regime_pct = 0.0

        # Step 4: Calculate Action Advantages (Counterfactuals)
        advantages = {
            int(a_idx): float(action_scores[a_idx] - chosen_score)
            for a_idx in range(nb_actions)
        }

        # Step 5: Build natural language explanation summary
        summary = self._generate_summary(
            action=action,
            action_label=action_meanings.get(action, f"Action {action}"),
            feature_names=feature_names,
            percentages=percentages,
            attributions=attributions,
            regime_pct=regime_pct,
            state=state,
        )

        return {
            "action": int(action),
            "action_label": action_meanings.get(action, f"Action {action}"),
            "feature_attributions": {
                name: float(attributions[i]) for i, name in enumerate(feature_names)
            },
            "feature_percentages": {
                name: float(percentages[i]) for i, name in enumerate(feature_names)
            },
            "regime_influence_score": float(regime_pct),
            "action_scores": {int(i): float(s) for i, s in enumerate(action_scores)},
            "action_advantages": advantages,
            "summary": summary,
        }

    def _evaluate_action_scores(self, agent: Any, state: np.ndarray) -> np.ndarray:
        """Extract action Q-values, policy logits, or probabilities for input state."""
        state_tensor_input = np.expand_dims(state, axis=0)

        # Try SB3 DQN policy q_net
        if hasattr(agent, "model"):
            agent = agent.model

        if hasattr(agent, "policy"):
            policy = agent.policy
            # SB3 DQN q_net
            if hasattr(policy, "q_net"):
                import torch

                with torch.no_grad():
                    obs_tensor = torch.as_tensor(state_tensor_input).to(policy.device)
                    q_vals = policy.q_net(obs_tensor).cpu().numpy().flatten()
                    return q_vals
            # SB3 PPO get_distribution
            elif hasattr(policy, "get_distribution"):
                import torch

                with torch.no_grad():
                    obs_tensor = torch.as_tensor(state_tensor_input).to(policy.device)
                    dist = policy.get_distribution(obs_tensor)
                    logits = dist.distribution.logits.cpu().numpy().flatten()
                    return logits

        # BaseAgentWrapper or custom wrapper fallback
        if hasattr(agent, "predict"):
            if hasattr(agent, "get_q_values"):
                return np.asarray(agent.get_q_values(state), dtype=np.float32)

            act = agent.predict(state)
            scores = np.zeros(4, dtype=np.float32)
            scores[act] = 1.0
            return scores

        return np.ones(4, dtype=np.float32) / 4.0

    def _generate_summary(
        self,
        action: int,
        action_label: str,
        feature_names: List[str],
        percentages: np.ndarray,
        attributions: np.ndarray,
        regime_pct: float,
        state: np.ndarray,
    ) -> str:
        """Synthesize a human-readable text explanation of the agent's action decision."""
        top_indices = np.argsort(percentages)[::-1][:2]
        top_drivers = [
            f"{feature_names[idx]} ({percentages[idx]:.1f}%)" for idx in top_indices
        ]
        drivers_str = " and ".join(top_drivers)

        regime_desc = ""
        if regime_pct > 10.0:
            regime_desc = (
                f" Market regime features contributed {regime_pct:.1f}% to this decision."
            )

        return (
            f"Policy selected '{action_label}'. "
            f"Key driving factors: {drivers_str}.{regime_desc}"
        )
