"""
Analysis & Regime Detection Script.

Fits MarketHMM on historical market features, performs causal online inference,
saves the causal regime dataset, and generates analysis plots.

Usage:
    python -m scripts.analyze_regimes
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.regimes.features import (
    RegimeFeatureEngine,
    RegimeFeatureScaler,
    chronological_split,
)
from src.regimes.hmm_model import MarketHMM
from src.regimes.inference import CausalRegimeInference
from src.regimes.evaluation import RegimeEvaluator


def main():
    print("=" * 70)
    print("  KAIRO: Market Regime Detection & Causal Inference Analysis")
    print("=" * 70)

    # 1. Load Data
    data_path = "data/processed/aapl_processed_20260830_20260904.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed market data not found at {data_path}")

    print(f"\n[1/6] Loading data from {data_path}...")
    raw_df = pd.read_parquet(data_path)
    print(f"      Loaded {len(raw_df)} bars from {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

    # 2. Compute Causal Features
    print("\n[2/6] Computing causal regime features...")
    engine = RegimeFeatureEngine()
    df_feat = engine.compute_features(raw_df)
    clean_df, X_all = engine.extract_feature_matrix(df_feat, drop_na=True)
    print(f"      Extracted feature matrix X of shape {X_all.shape} across {len(engine.feature_columns)} features:")
    for col in engine.feature_columns:
        print(f"        - {col}")

    # 3. Chronological Train / Validation / Test Split
    print("\n[3/6] Splitting data chronologically (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df = chronological_split(clean_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"      Train set: {len(train_df)} bars ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"      Val set:   {len(val_df)} bars ({val_df['timestamp'].min()} -> {val_df['timestamp'].max()})")
    print(f"      Test set:  {len(test_df)} bars ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")

    _, X_train = engine.extract_feature_matrix(train_df, drop_na=False)

    # 4. Fit HMM on Training Split ONLY
    print("\n[4/6] Training 4-State GaussianHMM on Training Split...")
    scaler = RegimeFeatureScaler(method="robust")
    hmm = MarketHMM(
        n_regimes=4,
        covariance_type="full",
        n_iter=150,
        random_state=42,
        feature_names=engine.feature_columns,
    )
    hmm.fit(X_train, scaler=scaler)

    print("      Model fitted successfully!")
    print("      Transition Matrix A:")
    print(np.round(hmm.transition_matrix, 4))
    print("      Stationary Distribution pi*:")
    print(np.round(hmm.stationary_distribution, 4))
    print("      Expected Regime Durations (bars):")
    for k, dur in hmm.expected_durations.items():
        print(f"        - Regime {k} ({hmm.regime_labels[k]}): {dur:.1f} bars")

    # Save trained model artifact
    model_save_path = "models/hmm_regime_model.joblib"
    os.makedirs("models", exist_ok=True)
    hmm.save(model_save_path)
    print(f"      Saved fitted HMM model to {model_save_path}")

    # 5. Causal Online Inference on Entire Dataset
    print("\n[5/6] Running Causal Forward Inference over full sequence...")
    inference_engine = CausalRegimeInference(hmm)
    causal_df = inference_engine.predict_causal(clean_df)

    # Save causal regime dataset
    out_parquet = "data/processed/aapl_regimes_causal.parquet"
    out_csv = "data/processed/aapl_regimes_causal.csv"
    causal_df.to_parquet(out_parquet)
    causal_df.to_csv(out_csv, index=False)
    print(f"      Saved causal regime dataset to:")
    print(f"        - {out_parquet}")
    print(f"        - {out_csv}")

    # 6. Evaluation & Plotting
    print("\n[6/6] Generating evaluation profiles & plots...")
    evaluator = RegimeEvaluator(hmm)
    summary_table = evaluator.generate_summary_table(causal_df)
    print("\n--- Empirical Regime Statistics Summary ---")
    print(summary_table.to_string(index=False))

    # Generate Visualizations
    os.makedirs("docs/images", exist_ok=True)
    _generate_plots(causal_df, hmm)
    print("\n[SUCCESS] Regime analysis complete! Plots saved to docs/images/")


def _generate_plots(df: pd.DataFrame, hmm: MarketHMM):
    sns.set_theme(style="darkgrid")
    regime_colors = {
        0: "#2ecc71",  # Low Vol - Emerald Green
        1: "#3498db",  # Normal - Blue
        2: "#f39c12",  # High Vol - Amber
        3: "#e74c3c",  # Stress - Red
    }

    # Figure 1: Price Chart with Regime Background Colors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    timestamps = pd.to_datetime(df["timestamp"])
    close = df["close"]
    regimes = df["causal_regime"]

    ax1.plot(timestamps, close, color="#2c3e50", lw=1.2, label="AAPL Close Price")
    ax1.set_title("AAPL Stock Price & Causally Inferred Market Regimes (HMM)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=12)

    # Highlight regime spans
    n = len(df)
    start_idx = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[start_idx]:
            r_id = regimes[start_idx]
            ax1.axvspan(
                timestamps.iloc[start_idx],
                timestamps.iloc[i - 1],
                color=regime_colors.get(r_id, "grey"),
                alpha=0.25,
                lw=0,
            )
            start_idx = i

    # Custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors[k], alpha=0.5, label=f"State {k}: {hmm.regime_labels[k]}")
        for k in range(hmm.n_regimes)
    ]
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True)

    # Subplot 2: Causal State Posterior Probabilities
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) == hmm.n_regimes:
        prob_matrix = df[prob_cols].to_numpy().T
        ax2.stackplot(
            timestamps,
            prob_matrix,
            labels=[hmm.regime_labels[k] for k in range(hmm.n_regimes)],
            colors=[regime_colors[k] for k in range(hmm.n_regimes)],
            alpha=0.7,
        )
        ax2.set_ylabel("Causal Prob", fontsize=11)
        ax2.set_ylim(0, 1.0)
        ax2.set_xlabel("Timestamp", fontsize=12)

    plt.tight_layout()
    fig_path = "docs/images/regimes_analysis.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"      Saved price regime chart to {fig_path}")

    # Figure 2: Transition Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    A = hmm.transition_matrix
    labels = [f"S{k}: {hmm.regime_labels[k]}" for k in range(hmm.n_regimes)]

    sns.heatmap(
        A,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
        vmin=0,
        vmax=1,
    )
    ax.set_title("HMM State Transition Matrix A", fontsize=12, fontweight="bold")
    ax.set_xlabel("To State (t)", fontsize=11)
    ax.set_ylabel("From State (t-1)", fontsize=11)
    plt.tight_layout()

    trans_path = "docs/images/transition_matrix.png"
    plt.savefig(trans_path, dpi=200)
    plt.close()
    print(f"      Saved transition matrix heatmap to {trans_path}")


if __name__ == "__main__":
    main()
