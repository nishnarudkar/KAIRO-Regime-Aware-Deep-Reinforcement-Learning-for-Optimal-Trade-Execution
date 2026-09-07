"""
Unit tests for Market Regime Detection module (HMM, Features, Causal Inference, Evaluation).
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.regimes.features import (
    RegimeFeatureEngine,
    RegimeFeatureScaler,
    chronological_split,
    DEFAULT_REGIME_FEATURES,
)
from src.regimes.hmm_model import MarketHMM, CANONICAL_REGIME_LABELS
from src.regimes.inference import CausalRegimeInference
from src.regimes.evaluation import RegimeEvaluator


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Generate synthetic OHLCV time series data for testing."""
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2026-01-01 09:30", periods=n, freq="1min")

    # Simulate price regime shifts
    returns = np.random.normal(0.0001, 0.001, n)
    returns[100:150] = np.random.normal(-0.0005, 0.005, n)[100:150]  # High vol / stress drop

    close = 150.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.001, n)))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, n)
    volume = np.random.uniform(1000, 50000, n)
    volume[100:150] *= 3.0  # Spike volume during stress

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_regime_feature_engine(sample_market_df):
    engine = RegimeFeatureEngine()
    df_feat = engine.compute_features(sample_market_df)

    for col in DEFAULT_REGIME_FEATURES:
        assert col in df_feat.columns

    clean_df, X = engine.extract_feature_matrix(df_feat, drop_na=True)
    assert len(clean_df) < len(sample_market_df)  # Warm-up dropped
    assert X.shape[1] == len(DEFAULT_REGIME_FEATURES)
    assert not np.isnan(X).any()


def test_chronological_split(sample_market_df):
    engine = RegimeFeatureEngine()
    df_feat = engine.compute_features(sample_market_df)
    clean_df, _ = engine.extract_feature_matrix(df_feat, drop_na=True)

    train_df, val_df, test_df = chronological_split(clean_df, 0.7, 0.15, 0.15)
    assert len(train_df) + len(val_df) + len(test_df) == len(clean_df)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_regime_feature_scaler(sample_market_df):
    engine = RegimeFeatureEngine()
    clean_df, X = engine.extract_feature_matrix(engine.compute_features(sample_market_df))

    scaler = RegimeFeatureScaler(method="robust")
    assert not scaler.is_fitted
    X_scaled = scaler.fit_transform(X)
    assert scaler.is_fitted
    assert X_scaled.shape == X.shape


def test_market_hmm_fitting_and_canonical_sorting(sample_market_df):
    engine = RegimeFeatureEngine()
    clean_df, X = engine.extract_feature_matrix(engine.compute_features(sample_market_df))

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42, feature_names=engine.feature_columns)
    hmm.fit(X, scaler=scaler)

    assert hmm.is_fitted
    assert hmm.transition_matrix.shape == (4, 4)
    # Row stochastic check
    np.testing.assert_allclose(hmm.transition_matrix.sum(axis=1), 1.0, atol=1e-5)

    # Stationary distribution check
    stat = hmm.stationary_distribution
    assert len(stat) == 4
    assert pytest.approx(stat.sum()) == 1.0

    # Durations check
    durations = hmm.expected_durations
    assert len(durations) == 4
    for k, dur in durations.items():
        assert dur >= 1.0


def test_model_persistence(sample_market_df, tmp_path):
    engine = RegimeFeatureEngine()
    clean_df, X = engine.extract_feature_matrix(engine.compute_features(sample_market_df))

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42, feature_names=engine.feature_columns)
    hmm.fit(X, scaler=scaler)

    save_file = tmp_path / "test_hmm.joblib"
    hmm.save(str(save_file))
    assert os.path.exists(save_file)

    loaded_hmm = MarketHMM.load(str(save_file))
    assert loaded_hmm.is_fitted
    assert loaded_hmm.n_regimes == 4
    np.testing.assert_allclose(hmm.transition_matrix, loaded_hmm.transition_matrix)


def test_causal_forward_inference_invariance(sample_market_df):
    """
    CRITICAL TEST:
    Verify that appending future data points at t+1, t+2 DOES NOT alter
    the causally inferred regime or posterior probability at timestamp t.
    """
    engine = RegimeFeatureEngine()
    clean_df, X = engine.extract_feature_matrix(engine.compute_features(sample_market_df))

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42, feature_names=engine.feature_columns)
    hmm.fit(X[:150], scaler=scaler)

    inference = CausalRegimeInference(hmm)

    # 1. Infer regime up to t = 100
    df_t100 = clean_df.iloc[:100].copy()
    res_t100 = inference.predict_causal(df_t100)
    state_at_t100_short = res_t100.iloc[99]["causal_regime"]
    prob_at_t100_short = res_t100.iloc[99]["prob_low_volatility"]

    # 2. Infer regime up to t = 200 (with 100 future observations appended)
    df_t200 = clean_df.iloc[:200].copy()
    res_t200 = inference.predict_causal(df_t200)
    state_at_t100_long = res_t200.iloc[99]["causal_regime"]
    prob_at_t100_long = res_t200.iloc[99]["prob_low_volatility"]

    # Invariant assertion: Future observations MUST NOT retroactively change state at t=100
    assert state_at_t100_short == state_at_t100_long
    assert pytest.approx(prob_at_t100_short, abs=1e-6) == prob_at_t100_long


def test_regime_evaluator(sample_market_df):
    engine = RegimeFeatureEngine()
    clean_df, X = engine.extract_feature_matrix(engine.compute_features(sample_market_df))

    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(n_regimes=4, random_state=42, feature_names=engine.feature_columns)
    hmm.fit(X, scaler=scaler)

    inference = CausalRegimeInference(hmm)
    df_regimes = inference.predict_causal(clean_df)

    evaluator = RegimeEvaluator(hmm)
    summary_df = evaluator.generate_summary_table(df_regimes)

    assert len(summary_df) == 4
    assert "Regime Name" in summary_df.columns
    assert "Frequency (%)" in summary_df.columns
