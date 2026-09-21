# Experimental Methodology

This document describes how KAIRO's research experiments are designed and how to reproduce them. The measured
results are in [`research_audit.md`](research_audit.md) and in `results/`.

---

## 1. Strategies evaluated

| Strategy | Type | State dim |
|---|---|---|
| TWAP | Baseline: equal slices | – |
| VWAP | Baseline: slices follow an ex-ante volume profile estimated from earlier days (flat = TWAP without history) | – |
| POV (10%) | Baseline: 10% of bar volume | – |
| DQN | RL agent, market state only | 7 |
| Regime-Aware DQN | DQN + causal HMM regime features | 12 |
| DQN (shuffled regime) | Ablation control: 12-dim state, regime features replaced by noise | 12 |
| PPO / Regime-Aware PPO / PPO (shuffled regime) | Same three variants for PPO | 7 / 12 / 12 |

---

## 2. Synthetic market

All data is **synthetic** and deterministic given a seed (`src/evaluation/scenarios.py`).

Each scenario is a one-minute market driven by a **latent Markov chain over four regimes** (low-vol, normal, high-vol,
stress). The regime sets the per-bar return volatility, the quoted spread and the traded volume:

| Regime | σ per minute | Mean spread | Volume multiplier |
|---|---|---|---|
| Low Volatility | 0.025% | $0.010 | 0.8 |
| Normal | 0.050% | $0.020 | 1.0 |
| High Volatility | 0.120% | $0.050 | 1.4 |
| Stress | 0.300% | $0.150 | 1.8 |

Design decisions that matter for validity:

* **Realistic scale.** An earlier version used 5% volatility *per minute*, so the price path dominated every result.
* **Zero drift** in all scenarios, so shortfall measures execution quality, not luck with the price path.
* **Bid and ask columns.** The simulator uses the data's quotes; previously the spread column was ignored and a
  constant 2 bps was used.
* **Causal volatility column** (20-bar backward-looking realized volatility) and trading-day timestamps
  (390 bars per day) with a mild intraday volume U-shape.
* **True regime** is stored in `true_regime`; agents and the HMM never see it. It is used only to validate the HMM.

| Scenario | Description |
|---|---|
| `normal` | Calm-to-normal, occasional high-vol spells |
| `high_volatility` | Mostly high-volatility regime |
| `low_liquidity` | Normal regimes, 25% volume, 3× spread |
| `stress` | Stress-dominated |
| `regime_transition` | Calm for the first 60% of the series, turbulent afterwards (the test region is entirely turbulent) |
| `liquidity_shock` | Normal, then volume collapses to 10% and spreads triple at 60% (the test region is entirely post-shock) |

---

## 3. Protocol (`src/evaluation/protocol.py`)

```
one series of 3000 bars per (scenario, seed)
  ├── train  (first 70%)  → HMM fit + agent training on RANDOM 30-bar windows
  └── test   (last 30%)   → 30 non-overlapping 30-bar windows, every strategy on the same windows
```

* **Chronological split, never shuffled.** The HMM is fitted on train features only.
* **Random-window training.** Each training episode starts at a random bar of the train region, so agents see
  thousands of distinct market paths. (Previously every episode replayed the same 30 bars.)
* **Paired evaluation.** All strategies run on identical test windows; comparisons are paired differences in
  implementation-shortfall bps (treatment − control, negative = cheaper).
* **Causal history.** The volume reference of the observation, the previous price, the VWAP profile and the
  HMM warm-up (30 bars) use only bars *before* the window start.
* **Environment.** Observations use fixed unit conversions (percent, bps, log-liquidity) so all features are O(1);
  rewards are expressed in bps of notional; impact is a calibrated square-root law (η = 0.5, γ = 0.001) giving a
  cost of roughly 10–30 bps for a 100k-share order.

### Statistics

For each comparison and scope (all scenarios, turbulent scenarios, each scenario) the suite reports:

* the **window level**: mean paired difference, bootstrap 95% CI, Wilcoxon signed-rank p-value and win rate over
  all test windows (windows of one seed share a trained agent, so treat p-values as indicative);
* the **seed level**: mean and spread of per-seed differences, and how many seeds favour the treatment. Seeds are
  the independent training replications.

No multiple-comparison correction is applied; with many comparisons, isolated small p-values should be treated with
caution.

---

## 4. Hyperparameters

Set in `src/evaluation/protocol.py::make_agent` and identical across all variants of an algorithm:

| | DQN | PPO |
|---|---|---|
| Network | MLP 128×128 | MLP 128×128 |
| Learning rate | 3e-4 | 3e-4 |
| Other | buffer 100k, ε 1.0→0.05 over 30% of training, target update 1000 | n_steps 1024, batch 128, 10 epochs |
| Training steps | 100,000 per model (research suite) / 200,000 (served models) | same |

These were fixed after a single pilot run on one scenario; no tuning was done on the test windows.

---

## 5. Reproducing

```bash
# Full suite: 6 scenarios × 8 seeds × 9 strategies, 100k steps per learned model (parallel workers)
python scripts/run_experiments.py --seeds 42 123 777 2024 31415 7 99 1234 --timesteps 100000 --jobs 16

# Quick smoke test
python scripts/run_experiments.py --scenarios normal --seeds 42 --timesteps 5000

# Train and save the models the API serves (200k steps each, held-out evaluation recorded in models/registry.json)
python scripts/train_models.py
```

Outputs in `results/`: `window_results.csv` (one row per window), `experiment_results.csv`, `summary_table.csv`,
`comparison_table.csv`, `per_scenario_table.csv`, `paired_comparisons.csv`, `hmm_validation.csv`,
`rq_summary.json`, `run_config.json`. Nothing is edited by hand.
