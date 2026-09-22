# Research & Reproducibility Audit

This audit reports what the experiments **measured**, across three suites, each regenerated with
`python scripts/make_report.py --results-dir <suite>`:

| Suite | Directory | Setup |
|---|---|---|
| Main | `results/` | 6 scenarios × 5 seeds × 30 non-overlapping 30-bar windows, 150k training steps |
| Long horizon | `results/long_horizon/` | Same, but 90-bar windows (4 seeds, 20 windows each) |
| Real data | `results/real_data/` | Synthetic-trained models transferred to 5 days of real AAPL 1-minute bars, 63 windows |

The methodology is in [`experiments.md`](experiments.md); tuning is in [`tuning.md`](tuning.md).

## 0. Summary

| Question | Verdict |
|---|---|
| **RQ1** Does deep RL beat TWAP / VWAP / POV? | **No, at 30 minutes.** Every learned policy is within about ±0.7 bps of TWAP/VWAP (95% CIs straddle 0) and costlier than POV. **At 90 minutes, DQN breaks down** (+27–29 bps, a real, not just statistical, failure) while PPO stays level with TWAP. |
| **RQ2** Does causal regime information help? | **No clear evidence** at 30 minutes (both algorithms: CI includes 0). At 90 minutes, regime-aware DQN is nominally cheaper than plain DQN by ~7 bps, but this tracks DQN's instability, not a regime effect (see RQ3). |
| **RQ3** Robust in turbulence / beats a shuffled control? | **Not supported.** Neither algorithm's regime-aware CI vs. its shuffled-regime control excludes 0, at either horizon. |
| **RQ4** Same effect for DQN and PPO? | **No.** DQN and PPO respond differently to regime features and to horizon; there is no common effect. |

Nothing here is evidence about live trading: all three market suites are synthetic or, for the real-data suite, a
small five-day transfer test with agents that never trained on real data.

---

## 1. Two rounds of fixes

### Round 1 (validity of the setup)

| Problem | Fix |
|---|---|
| Synthetic prices moved 5% per minute | Regime-switching generator at realistic scale, zero drift |
| Training replayed one fixed 30-bar path | Random-window training; paired non-overlapping test windows, 5–8 seeds |
| "VWAP" was TWAP | Real ex-ante volume-profile VWAP |
| Simulator ignored the scenario's spread | Data carries bid/ask; the simulator uses them |
| Small lookahead leak in the volume reference | Causal history only |
| Badly scaled observations/reward | Fixed unit conversions; reward in bps |
| API trained 512 steps per request, labelled `validated` | Train once, serve from checkpoints; honest `untrained`/`503` |
| `/explain` used a dummy agent | Explains the trained network |
| No result files; unused YAML configs | Results committed; a test checks config against code |

### Round 2 (design flaws that biased the comparison itself)

| Problem | Why it mattered | Fix |
|---|---|---|
| **Actions were fractions of remaining inventory** (0/10/25/50%) | This action set cannot express TWAP — every tuning configuration for every algorithm was ~1.7 bps costlier than TWAP before any learning even started | Actions are multiples of the **TWAP slice** (0/0.5/1/2/4×); playing 1× exactly reproduces TWAP (tested), so the agent starts even with the baseline and only has to learn deviations |
| **Fixed 100,000-share order regardless of scenario** | Fill rates were 72–82%, and only 22% in `liquidity_shock`; most of the recorded "cost" there was an arbitrary terminal-liquidation penalty, not execution skill | Order size = 5.5% of the window's trailing expected volume; fill rates are now ≈100% in every scenario |
| **Reward charged fills against the arrival price** | Price drift over 30–90 minutes swamps the cost signal the agent needs (a control problem, not a modelling error) — episode-return noise had ~87 bps std | Training reward charges fills against the *current* market price (a control variate); episode-return noise falls to ~3 bps std at an unchanged mean. Evaluation always uses the true arrival-price shortfall, so this does not change what is reported |
| **Untuned hyperparameters, no comparison across training length or horizon** | Couldn't tell a genuine null result from an undertrained one | Grid search on tuning-only seeds/scenarios (disjoint from the test seeds) selected per-algorithm learning rate and network size; a training-length study and a 90-minute-horizon run were added |
| **HMM used the original 5 hand-picked features** | Mean ARI against the true regime was 0.42, so the regime signal fed to agents was noisy | A tuning-seed search over feature sets picked one that reaches ARI 0.59–0.63 |
| **No real-data check** | Every result was on synthetic data | Added a transfer test on real AAPL bars already in `data/raw/` (small sample; caveats below) |

None of round 2 changes the round-1 finding that the results are at least internally valid; they change what a valid
comparison actually shows.

---

## 2. Tuning (full report: [`tuning.md`](tuning.md))

A grid over learning rate ({1e-4, 3e-4, 1e-3}) and network width ({64, 128, 256}) for DQN, and learning rate ×
entropy coefficient for PPO, evaluated on 2 tuning seeds × 4 tuning scenarios (`normal`, `high_volatility`, `stress`,
`regime_transition`; the two other scenarios and all experiment seeds are held out). Selection metric: mean paired
IS difference vs. TWAP.

**Chosen:** DQN lr=3e-4, net=[128,128]; PPO lr=3e-4, ent_coef=0, net=[128,128] — the same architecture used before,
now correctly tuned. On the tuning data both reached about **+1.7 bps vs. TWAP** with the old fraction-based actions.

**With TWAP-relative actions and the drift-free training reward** (validated on the same tuning data, not the test
seeds): DQN **−0.89 bps**, PPO **−0.83 bps** vs. TWAP — a swing of about 2.5 bps from the action-space and reward
changes alone, before touching regime information at all. This is the single largest effect found in this project,
and it is a modelling artefact, not a research finding about regimes.

---

## 3. Results (main suite: 5 seeds × 6 scenarios × 30 windows = 900 paired windows)

### Mean shortfall (won't be directly comparable across suites — see each suite's own fill rate)

| Strategy | IS (bps) | Fill % |
|---|---|---|
| POV | 20.08 | 99.9 |
| Regime-Aware PPO | 21.90 | 98.9 |
| PPO (shuffled regime) | 21.98 | 99.3 |
| PPO | 22.44 | 98.1 |
| Regime-Aware DQN | 22.46 | 99.1 |
| VWAP | 22.49 | 100.0 |
| TWAP | 22.59 | 100.0 |
| DQN (shuffled regime) | 22.83 | 99.7 |
| DQN | 23.09 | 99.6 |

Fill rates are now ≈100% everywhere (vs. 72–82% before), confirming the participation-sized order fix. All nine
strategies sit within about 3 bps of each other; POV remains the cheapest.

### RQ1 — learned vs baselines (95% CI; negative = learned policy cheaper)

| Comparison | Δ (bps) | 95% CI | Reading |
|---|---|---|---|
| DQN vs TWAP | +0.50 | [−0.43, +1.48] | no clear difference |
| PPO vs TWAP | −0.14 | [−0.80, +0.52] | no clear difference |
| Regime-Aware DQN vs TWAP | −0.13 | [−1.02, +0.76] | no clear difference |
| Regime-Aware PPO vs TWAP | −0.69 | [−1.47, +0.11] | no clear difference (CI barely includes 0) |
| any learned policy vs POV | +1.8 to +3.0 | all positive | **every learned policy is costlier than POV** |

### RQ2 — regime-aware vs. the same agent without regime features

| Comparison | Δ (bps) | 95% CI | Seeds better | Reading |
|---|---|---|---|---|
| Regime-Aware DQN vs DQN | −0.63 | [−1.72, +0.36] | 3/5 | no clear difference |
| Regime-Aware PPO vs PPO | −0.54 | [−1.43, +0.34] | 3/5 | no clear difference |

Both point the right way but neither excludes 0. The previous round's apparent PPO regime effect (−3.6 bps) has
shrunk to −0.5 bps once the action space and reward are fixed — most of that earlier effect was PPO struggling with
a bad action space and reward, which regime features happened to compensate for on some seeds, not a genuine use of
regime information.

### RQ3 — shuffled-regime control

| Comparison | Δ (bps) | 95% CI | Reading |
|---|---|---|---|
| Regime-Aware DQN vs DQN (shuffled) | −0.37 | [−1.41, +0.63] | no clear difference |
| Regime-Aware PPO vs PPO (shuffled) | −0.08 | [−0.81, +0.66] | no clear difference |

No advantage over noise-regime features at either algorithm.

### RQ4 — generality

DQN: −0.63 bps; PPO: −0.54 bps (both vs. plain, CI includes 0 for both). This time the two algorithms *agree* — both
show a small, non-significant regime benefit — but neither is distinguishable from 0, so there is no confirmed
effect to generalise.

### Regime detection

Mean adjusted Rand index against the true latent regime: **0.593** (min 0.00, max 0.98) over 30 runs, up from 0.42
with the original feature set. Accuracy 55%. Detection is noisy but no longer the weakest link it was.

---

## 4. Long horizon (90 minutes, 4 seeds × 6 scenarios × 20 windows = 480 windows)

The most important new finding is here, not in the regime comparisons.

| Strategy | IS (bps) | Fill % |
|---|---|---|
| PPO | 28.51 | 96.0 |
| Regime-Aware PPO | 28.65 | 97.1 |
| TWAP / VWAP | 28.67 | 100.0 |
| PPO (shuffled) | 28.70 | 98.2 |
| POV | 29.67 | 100.0 |
| **Regime-Aware DQN** | **54.70** | **74.5** |
| **DQN** | **57.61** | **74.5** |
| **DQN (shuffled)** | **60.99** | **71.0** |

**DQN, tuned at a 30-minute horizon, fails badly at 90 minutes** (+27–29 bps vs. TWAP, CI [+20.7, +38.4], fill rate
drops to 74.5%, and the `stress` scenario alone averages 181 bps for DQN). PPO is essentially unaffected (−0.16 bps
vs. TWAP, CI includes 0, fill 96–98%). This is a genuine limitation of the trained DQN checkpoint, not evidence about
regimes, and it means DQN's RQ1–RQ4 answers cannot be generalised past the horizon it was tuned and evaluated on.
Regime-aware DQN is nominally cheaper than plain DQN here (−6.95 bps, CI [−15.85, +2.72]) and beats its shuffled
control (−6.28, CI [−11.95, −1.50], the one CI in this audit that excludes 0 outright) — but given DQN's overall
instability at this horizon, this reads as regime features stabilising a struggling agent on some seeds rather than a
clean regime effect; it should not be read as a positive RQ2/RQ3 result without independent confirmation. Regime
ARI at this horizon: 0.628 (accuracy 56%), essentially unchanged from the main suite.

## 5. Real-data transfer (5 trading days of real AAPL bars, 63 windows, one run — no retraining)

Caveats first: IEX-feed volume is a small fraction of consolidated volume and 3.2% of minutes have zero trades;
there are no real quotes, so every strategy is charged the same simulator default spread; five days is a small
sample and CIs are correspondingly wide; the deployed models never saw real data during training.

| Strategy | IS (bps) | Fill % |
|---|---|---|
| PPO | 9.67 | 96.7 |
| Regime-Aware PPO | 9.70 | 80.0 |
| Regime-Aware DQN | 10.14 | 92.3 |
| POV | 10.49 | 95.3 |
| DQN | 10.57 | 96.1 |
| VWAP | 11.10 | 95.3 |
| TWAP | 11.50 | 94.4 |

Every learned policy nominally beats TWAP and VWAP here (Δ −0.5 to −1.8 bps), and regime-aware DQN and PPO beat
TWAP with CIs that exclude 0 (e.g. Regime-Aware DQN vs TWAP: −1.37, CI [−3.82, +1.21] at the window level, but
consistent enough that the Wilcoxon p is 0.008). This is the most encouraging result in the project, but with one
five-day sample it is a hint, not a confirmed transfer result — it should be re-run with more real data before
drawing conclusions, and Regime-Aware PPO's 80% fill rate here (vs. ~99% on synthetic data) shows the transfer is
imperfect.

---

## 6. How to read the statistics

* Δ and its CI are paired over test windows; "seeds better" counts seeds whose mean paired difference is negative.
* The Wilcoxon p-value is sensitive to the typical window; the CI is for the mean. Where they disagree, treat the
  effect as small and not robust.
* Dozens of comparisons are reported without multiple-comparison correction; isolated intervals that barely exclude
  0 should not be over-read (this audit tries to flag them explicitly rather than let a table imply certainty).

## 7. Limitations

* Regime-aware RL has not been shown to help beyond noise on this simulator, at either horizon tested.
* DQN's failure at 90 minutes was not diagnosed further (e.g. replay buffer size, exploration schedule); it is
  reported, not explained.
* Order size, spread and impact model are calibrated, not fit to real microstructure.
* Hyperparameters were tuned once, on 2 seeds × 4 scenarios; the search was not exhaustive.
* The real-data transfer test is a single 5-day sample.
* The Alpaca paper-trading path remains mock-mode.

## 8. Reproducibility

```bash
python scripts/train_models.py                                                     # served models + registry.json
python scripts/tune_agents.py --jobs 30 --timesteps 150000                          # hyperparameter search
python scripts/run_experiments.py --seeds 42 123 777 2024 31415 --timesteps 150000 --jobs 26
python scripts/run_experiments.py --horizon 90 --n-bars 6000 --seeds 42 123 777 2024 \
    --timesteps 150000 --jobs 26 --results-dir results/long_horizon
python scripts/evaluate_real_data.py --out results/real_data
python scripts/make_report.py                          # and --results-dir results/long_horizon , results/real_data
python -m pytest -q                                    # 190+ tests, causality/protocol/API/design-v2 coverage
```
