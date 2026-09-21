"""
Causal and Online Regime Inference Engine.

CRITICAL LEAKAGE PREVENTION PROTOCOL:
Standard HMM decoding (Viterbi or smoothing) uses the FULL sequence X_{1:T} to infer
state S_t at time t. This is NON-CAUSAL as S_t depends on future observations X_{t+1:T}.

This module implements STRICT CAUSAL FILTERING via the HMM Forward Algorithm:
  P(S_t = j | X_{1:t}) = normalize( b_j(X_t) * sum_i [ P(S_{t-1} = i | X_{1:t-1}) * A_{ij} ] )

At timestamp t:
1. Only data X_1, ..., X_t is consumed.
2. The belief state update is strictly forward recursive.
3. Future data X_{t+1:T} is completely hidden and has zero influence on S_t.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from src.regimes.hmm_model import MarketHMM


class CausalRegimeInference:
    """
    Causal / Online Regime Inference Engine using HMM Forward Filtering.
    """

    def __init__(self, hmm_model: MarketHMM):
        """
        Args:
            hmm_model: Fitted MarketHMM instance.
        """
        if not hmm_model.is_fitted or hmm_model.model is None:
            raise RuntimeError("MarketHMM must be fitted before initializing CausalRegimeInference.")

        self.hmm = hmm_model
        self.n_regimes = hmm_model.n_regimes
        self.feature_names = hmm_model.feature_names
        self.regime_labels = hmm_model.regime_labels

        # Model parameters
        self.A = hmm_model.model.transmat_  # Shape (K, K)
        self.means = hmm_model.model.means_  # Shape (K, D)
        self.covars = hmm_model.model.covars_  # Shape depends on covariance_type
        self.startprob = hmm_model.model.startprob_  # Shape (K,)
        self.cov_type = hmm_model.covariance_type

        # Streaming state
        self.current_belief: Optional[np.ndarray] = None

    def reset_online_state(self) -> None:
        """Reset streaming online state buffer."""
        self.current_belief = None

    def _prepare_gaussians(self) -> None:
        """Precompute precision matrices and log-determinants for fast per-step evaluation."""
        K = self.n_regimes
        D = self.means.shape[1]
        self._precisions = np.zeros((K, D, D))
        self._log_norm = np.zeros(K)
        for k in range(K):
            if self.cov_type == "full":
                cov_k = np.asarray(self.covars[k])
            elif self.cov_type == "diag":
                cov_k = np.diag(self.covars[k])
            elif self.cov_type == "spherical":
                cov_k = np.eye(D) * float(np.ravel(self.covars[k])[0])
            else:
                cov_k = np.asarray(self.covars[k])
            cov_k = cov_k + np.eye(D) * 1e-6   # tiny diagonal regularisation
            sign, logdet = np.linalg.slogdet(cov_k)
            self._precisions[k] = np.linalg.pinv(cov_k)
            self._log_norm[k] = -0.5 * (D * np.log(2.0 * np.pi) + logdet)

    def _emission_log_likelihoods(self, x: np.ndarray) -> np.ndarray:
        """
        Compute emission log likelihoods ln P(x | S = k) for observation vector x (D,).

        Returns:
            1D array of shape (K,) containing log probabilities.
        """
        if not hasattr(self, "_precisions"):
            self._prepare_gaussians()
        diff = x[None, :] - self.means                      # (K, D)
        maha = np.einsum("kd,kde,ke->k", diff, self._precisions, diff)
        return self._log_norm - 0.5 * maha

    def step_online(self, x_t: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Perform a single causal forward update step given feature vector x_t at time t.
        
        Args:
            x_t: Unscaled or scaled feature vector of shape (D,).
                 If scaler exists on HMM, x_t will be scaled inside.
                 
        Returns:
            Tuple of (inferred_regime_id, regime_posterior_probabilities_vector).
        """
        if x_t.ndim == 2:
            x_t = x_t.ravel()

        if self.hmm.scaler is not None:
            x_t_scaled = self.hmm.scaler.transform(x_t.reshape(1, -1)).ravel()
        else:
            x_t_scaled = x_t

        log_b = self._emission_log_likelihoods(x_t_scaled)
        # Convert log likelihoods safely using log-sum-exp
        max_log = np.max(log_b)
        b = np.exp(log_b - max_log)

        if self.current_belief is None:
            # t = 1: Prior * Emission
            unnorm = self.startprob * b
        else:
            # t > 1: (Belief_{t-1} @ A) * Emission
            predicted_prior = self.current_belief @ self.A
            unnorm = predicted_prior * b

        sum_unnorm = np.sum(unnorm)
        if sum_unnorm <= 0 or np.isnan(sum_unnorm):
            # Fallback to uniform prior if underflow occurs
            belief_t = np.ones(self.n_regimes) / self.n_regimes
        else:
            belief_t = unnorm / sum_unnorm

        self.current_belief = belief_t
        regime_id = int(np.argmax(belief_t))
        return regime_id, belief_t

    def predict_causal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch causal forward pass over historical DataFrame.
        
        Guarantees that at index t, regime assignment depends ONLY on observations 0..t.
        
        Args:
            df: Feature DataFrame containing required feature columns.
            
        Returns:
            DataFrame with original columns plus:
              - 'causal_regime': discrete regime ID (0..K-1)
              - 'causal_regime_name': human readable label
              - 'regime_prob_0' .. 'regime_prob_{K-1}': soft belief posterior probabilities
        """
        from src.regimes.features import RegimeFeatureEngine

        engine = RegimeFeatureEngine(feature_columns=self.feature_names)
        clean_df, X = engine.extract_feature_matrix(df, drop_na=True)

        n_samples = len(X)
        regimes = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, self.n_regimes), dtype=np.float64)

        self.reset_online_state()

        for t in range(n_samples):
            reg_id, belief = self.step_online(X[t])
            regimes[t] = reg_id
            probs[t] = belief

        out_df = clean_df.copy()
        out_df["causal_regime"] = regimes
        out_df["causal_regime_name"] = [
            self.regime_labels.get(r, f"Regime_{r}") for r in regimes
        ]

        for k in range(self.n_regimes):
            label_slug = self.regime_labels.get(k, f"regime_{k}").lower().replace(" ", "_")
            out_df[f"prob_{label_slug}"] = probs[:, k]

        return out_df
