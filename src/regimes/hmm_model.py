"""
Gaussian Hidden Markov Model (HMM) for Market Regime Detection.

Provides:
- Unsupervised learning of latent states via hmmlearn GaussianHMM
- Canonical state sorting (Low Vol -> Normal -> High Vol -> Stress) for stable interpretability
- Transition matrix, stationary distribution, and persistence metrics
- Model serialization (save / load with scaler and metadata)
"""

from typing import List, Optional, Dict, Any, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from src.regimes.features import RegimeFeatureScaler, DEFAULT_REGIME_FEATURES


CANONICAL_REGIME_LABELS = {
    0: "Low Volatility",
    1: "Normal",
    2: "High Volatility",
    3: "Stress",
}


class MarketHMM:
    """
    Market Regime Hidden Markov Model.
    
    Models market dynamics as a first-order Markov process with latent discrete
    regimes emitting multivariate Gaussian feature observations.
    
    CRITICAL METHODOLOGY:
    Hidden states in HMMs are latent and initially unidentifiable (label switching).
    After fitting, states are canonicalized by sorting in ascending order of their
    observed volatility metric. This guarantees:
      - State 0: Low Volatility
      - State 1: Normal
      - State 2: High Volatility
      - State 3: Stress / Extreme Dynamics
    """

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
        regime_labels: Optional[Dict[int, str]] = None,
        min_covar: float = 1e-3,
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_covar = min_covar
        self.feature_names = feature_names or DEFAULT_REGIME_FEATURES
        self.regime_labels = regime_labels or {
            i: CANONICAL_REGIME_LABELS.get(i, f"Regime_{i}") for i in range(n_regimes)
        }

        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[RegimeFeatureScaler] = None
        self.is_fitted: bool = False
        self._state_permutation: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        scaler: Optional[RegimeFeatureScaler] = None,
        volatility_col_idx: Optional[int] = None,
    ) -> "MarketHMM":
        """
        Fit GaussianHMM on training feature matrix.
        
        Args:
            X_train: 2D array of shape (N, D) of training features (already scaled or unscaled).
            scaler: Optional RegimeFeatureScaler. If provided, X_train is scaled using this scaler.
            volatility_col_idx: Index of feature column used for canonical state sorting.
                                If None, defaults to finding 'realized_vol' or 'parkinson_vol'
                                in feature_names, or index 1.
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2-dimensional (samples, features), got shape {X_train.shape}")
        if X_train.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_names)} features, got {X_train.shape[1]}"
            )

        self.scaler = scaler
        if self.scaler is not None:
            if not self.scaler.is_fitted:
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
        else:
            X_train_scaled = X_train

        # Initialize GaussianHMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            min_covar=self.min_covar,
        )

        # Fit model parameters using Baum-Welch (EM) algorithm
        self.model.fit(X_train_scaled)

        # Reorder states canonically
        self._canonicalize_states(X_train_scaled, volatility_col_idx=volatility_col_idx)
        self.is_fitted = True
        return self

    def _canonicalize_states(
        self,
        X: np.ndarray,
        volatility_col_idx: Optional[int] = None
    ) -> None:
        """
        Sort hidden states in ascending order of volatility so that:
        State 0 = Lowest volatility, State K-1 = Highest volatility/stress.
        
        Reorders means_, covars_, transmat_, and startprob_ in-place.
        """
        if self.model is None:
            return

        # Determine sorting column
        if volatility_col_idx is None:
            for idx, name in enumerate(self.feature_names):
                if "realized_vol" in name or "parkinson" in name or "vol" in name:
                    volatility_col_idx = idx
                    break
            if volatility_col_idx is None:
                volatility_col_idx = 0

        # Sort states by the mean value of the selected volatility feature
        vol_means = self.model.means_[:, volatility_col_idx]
        order = np.argsort(vol_means)
        self._state_permutation = order

        # Reorder parameters
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.means_ = self.model.means_[order]

        if self.covariance_type == "full":
            covs = self.model.covars_[order].copy()
            for i in range(self.n_regimes):
                # Enforce symmetry and positive-definiteness
                covs[i] = 0.5 * (covs[i] + covs[i].T) + np.eye(covs[i].shape[0]) * 1e-4
            self.model.covars_ = covs
        elif self.covariance_type in ("diag", "spherical"):
            covs = self.model.covars_[order].copy()
            covs = np.maximum(covs, 1e-4)
            self.model.covars_ = covs

    @property
    def transition_matrix(self) -> np.ndarray:
        """State transition probability matrix A where A[i, j] = P(S_t=j | S_{t-1}=i)."""
        self._check_fitted()
        return self.model.transmat_

    @property
    def stationary_distribution(self) -> np.ndarray:
        """
        Stationary distribution pi* satisfying pi* A = pi* and sum(pi*) = 1.
        """
        self._check_fitted()
        A = self.model.transmat_
        # Solve (A^T - I) pi = 0 subject to sum(pi) = 1
        n = A.shape[0]
        M = A.T - np.eye(n)
        M[-1, :] = 1.0
        rhs = np.zeros(n)
        rhs[-1] = 1.0
        try:
            pi = np.linalg.solve(M, rhs)
            pi = np.clip(pi, 0.0, 1.0)
            return pi / np.sum(pi)
        except np.linalg.LinAlgError:
            # Fallback: power iteration of transition matrix
            pi = np.ones(n) / n
            for _ in range(500):
                pi = pi @ A
            return pi

    @property
    def expected_durations(self) -> Dict[int, float]:
        """
        Expected duration in time steps for each regime: d_i = 1 / (1 - A_ii).
        """
        self._check_fitted()
        durations = {}
        for i in range(self.n_regimes):
            diag = float(self.model.transmat_[i, i])
            dur = 1.0 / (1.0 - diag) if diag < 1.0 else float("inf")
            durations[i] = dur
        return durations

    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood of observation sequence under the model."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        return float(self.model.score(X_scaled))

    def aic(self, X: np.ndarray) -> float:
        """
        Akaike Information Criterion (AIC):
        AIC = 2k - 2 ln(L)
        where k is number of free parameters and L is likelihood.
        """
        self._check_fitted()
        ll = self.score(X)
        k = self._count_parameters()
        return 2 * k - 2 * ll

    def bic(self, X: np.ndarray) -> float:
        """
        Bayesian Information Criterion (BIC):
        BIC = k ln(N) - 2 ln(L)
        """
        self._check_fitted()
        ll = self.score(X)
        n_samples = len(X)
        k = self._count_parameters()
        return k * np.log(n_samples) - 2 * ll

    def _count_parameters(self) -> int:
        """Count number of free parameters in GaussianHMM."""
        n_comp = self.n_regimes
        n_feat = len(self.feature_names)

        # Transition matrix: n_comp * (n_comp - 1)
        k_trans = n_comp * (n_comp - 1)
        # Initial probabilities: n_comp - 1
        k_start = n_comp - 1
        # Means: n_comp * n_feat
        k_means = n_comp * n_feat
        # Covariances
        if self.covariance_type == "full":
            k_covar = n_comp * n_feat * (n_feat + 1) // 2
        elif self.covariance_type == "diag":
            k_covar = n_comp * n_feat
        elif self.covariance_type == "spherical":
            k_covar = n_comp
        else:
            k_covar = n_comp

        return k_trans + k_start + k_means + k_covar

    def save(self, filepath: str) -> None:
        """Serialize trained model, scaler, and metadata to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "n_regimes": self.n_regimes,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "min_covar": self.min_covar,
            "feature_names": self.feature_names,
            "regime_labels": self.regime_labels,
            "state_permutation": self._state_permutation,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MarketHMM":
        """Deserialize trained model bundle from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        bundle = joblib.load(filepath)
        instance = cls(
            n_regimes=bundle["n_regimes"],
            covariance_type=bundle["covariance_type"],
            n_iter=bundle["n_iter"],
            random_state=bundle["random_state"],
            min_covar=bundle["min_covar"],
            feature_names=bundle["feature_names"],
            regime_labels=bundle["regime_labels"],
        )
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance._state_permutation = bundle.get("state_permutation")
        instance.is_fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MarketHMM is not fitted. Call fit() or load() first.")
