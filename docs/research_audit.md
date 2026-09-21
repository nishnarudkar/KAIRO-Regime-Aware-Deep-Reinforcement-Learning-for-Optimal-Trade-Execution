# Research & Reproducibility Audit

This audit reports what the experiments **measured**. All numbers below come from `results/` and are regenerated with
`python scripts/make_report.py` (full tables in [`results/REPORT.md`](../results/REPORT.md)). The methodology is in
[`experiments.md`](experiments.md).

## 0. Summary

| Question | Verdict from the data |
|---|---|
| **RQ1** Does deep RL beat TWAP / VWAP / POV? | **Not robustly.** Regime-aware DQN and PPO beat TWAP by ~1.6–1.8 bps (of ~23), plain DQN and PPO are not distinguishable from TWAP, and **no learned policy beats POV**, the best baseline. |
| **RQ2** Does causal regime information help? | **No clear evidence.** DQN: −0.25 bps, CI includes 0. PPO: −3.6 bps, but the shuffled-regime control shows the gain is not from regime information (see RQ3). |
| **RQ3** Robust in turbulence / beats a shuffled control? | **Not supported.** Regime-aware PPO does not beat its shuffled control (+0.48 bps, CI includes 0). Regime-aware DQN beats its control by 0.70 bps (CI −1.3 to −0.1, 6/8 seeds) — weak, borderline, and not confirmed by the rank test. |
| **RQ4** Same effect for DQN and PPO? | **No.** The two algorithms disagree, and neither shows a regime effect that survives the control. |

The previous version of this document reported "YES" for all four questions with 15–35% improvements. Those
statements were not backed by any result files, and the setup that would have produced them was invalid (Section 1).
They are withdrawn.

---

## 1. Problems found in the original setup (all fixed)

| Problem | Effect | Fix |
|---|---|---|
| Synthetic prices had **5% volatility per minute** | Implementation shortfall was dominated by the random price path (an agent that waited looked brilliant) | Regime-switching generator at realistic scale (0.02–0.3% per minute), zero drift |
| Scenario `spread` column was ignored by the simulator (constant 2 bps) | Scenario spreads never affected cost | Generator emits `bid`/`ask`, the simulator uses them |
| Every training episode replayed the **same 30 bars**; evaluation was one 30-bar path | Agents memorised one path; no statistics possible | Random-window training; 30 paired test windows per (scenario, seed), 8 seeds |
| Volume normaliser used the whole series mean | Small lookahead leak | Causal history only |
| Raw observations spanned 1e-3 … 1e6; rewards ~1e-4 | Poor conditions for gradient learning | Fixed unit scaling, reward in bps |
| **"VWAP" was TWAP** (the volume estimate was computed and ignored) | VWAP comparisons meaningless | Real ex-ante volume-profile VWAP |
| Regime filter started cold with NaN→0 features each episode | Bad regime input | Causal warm-start over preceding bars |
| API trained a fresh agent for 512 steps **per request**; models reported `validated` | The UI showed near-untrained policies as validated results | Train once, serve from checkpoints; honest `untrained` status; `503` when missing |
| `/explain` used a hand-written dummy agent; feature/action labels did not match the environment | Explanations were fiction | Explains the trained network; labels match the environment |
| No result files, YAML configs not used by code | Claims unverifiable | Result files committed; config/code consistency test |

---

## 2. Results

Setup: 6 scenarios × 8 seeds × 30 paired test windows = 1,440 windows per strategy; 100,000 training steps per learned
model; 100,000-share buy over 30 minutes. Δ = treatment − control in implementation-shortfall (IS) bps; negative
means the treatment is cheaper.

### Mean shortfall (all scenarios, seeds, windows)

| Strategy | IS (bps) | Fill % |
|---|---|---|
| POV | 20.09 | 77.1 |
| PPO (shuffled regime) | 20.61 | 81.4 |
| Regime-Aware PPO | 21.09 | 80.9 |
| Regime-Aware DQN | 21.25 | 72.1 |
| DQN | 21.50 | 75.0 |
| DQN (shuffled regime) | 21.95 | 71.4 |
| VWAP | 22.74 | 82.1 |
| TWAP | 22.83 | 82.0 |
| PPO | 24.70 | 77.3 |

Fill rates are below 100% because of the 15%-of-volume participation cap, which cannot fill a 100k-share order in
the thin `low_liquidity` and `liquidity_shock` scenarios; unfilled shares are charged a terminal liquidation penalty
that is included in IS. Differences between strategies are small (about 1–2 bps, and more for plain PPO) relative to
the spread of individual windows (tens of bps).

### RQ1 — learned policies vs baselines

| Comparison | Δ IS (bps) | 95% CI | Seeds better |
|---|---|---|---|
| DQN vs TWAP | −1.33 | [−2.89, +0.21] | 5/8 |
| Regime-Aware DQN vs TWAP | −1.58 | [−3.18, −0.03] | 6/8 |
| PPO vs TWAP | +1.87 | [−0.00, +3.86] | 2/8 |
| Regime-Aware PPO vs TWAP | −1.75 | [−3.36, −0.03] | 6/8 |
| DQN vs POV | +1.42 | [+0.41, +2.44] | 2/8 |
| Regime-Aware DQN vs POV | +1.17 | [+0.23, +2.11] | 2/8 |
| PPO vs POV | +4.62 | [+3.01, +6.29] | 2/8 |
| Regime-Aware PPO vs POV | +1.00 | [+0.23, +1.83] | 3/8 |

Against the equal-slice baselines the regime-aware agents are marginally cheaper (their CIs end just below 0). Against
POV, which is the strongest baseline here, every learned policy is costlier.

### RQ2 — regime-aware vs the same agent without regime features

| Scope | Comparison | Δ IS (bps) | 95% CI | Seeds better |
|---|---|---|---|---|
| All | Regime-Aware DQN vs DQN | −0.25 | [−0.83, +0.30] | 6/8 |
| All | Regime-Aware PPO vs PPO | −3.61 | [−5.26, −2.06] | 7/8 |
| Turbulent* | Regime-Aware DQN vs DQN | −0.41 | [−1.17, +0.33] | 6/8 |
| Turbulent* | Regime-Aware PPO vs PPO | −5.49 | [−7.95, −3.11] | 7/8 |

\* `regime_transition`, `stress`, `high_volatility`, `liquidity_shock`.

The PPO gain is concentrated in `regime_transition`, where plain PPO averages 39.2 bps against 19.8 (regime-aware) and
17.5 (shuffled-regime). Plain PPO does much worse there, but the gap is not consistent across seeds (only 4/8 seeds
favour regime-aware in that scenario), which points to training variance rather than a systematic regime effect.

### RQ3 — the shuffled-regime control

| Scope | Comparison | Δ IS (bps) | 95% CI | Seeds better |
|---|---|---|---|---|
| All | Regime-Aware DQN vs DQN (shuffled) | −0.70 | [−1.32, −0.12] | 6/8 |
| All | Regime-Aware PPO vs PPO (shuffled) | +0.48 | [−0.26, +1.17] | 2/8 |
| Turbulent | Regime-Aware DQN vs DQN (shuffled) | −0.90 | [−1.78, −0.07] | 4/8 |
| Turbulent | Regime-Aware PPO vs PPO (shuffled) | +0.72 | [−0.28, +1.76] | 2/8 |

The control does its job: it shows the PPO improvement in RQ2 is reproduced by an agent given *random* regime
features, so it is not evidence that regime information is useful. For DQN there is a small, borderline advantage over
the control.

### RQ4 — generality across algorithms

DQN and PPO do not agree (−0.25 vs −3.61 bps against their plain versions; −0.70 vs +0.48 against their controls).
There is no algorithm-independent regime effect in these data.

### Regime detection

Against the true latent regime on the test region, the causal HMM reaches a mean adjusted Rand index of 0.415
(min −0.001, max 0.788) and 55% accuracy over 48 runs. That is only moderate agreement, so the regime signal handed
to the agents is noisy (a per-regime confusion analysis was not performed).

### Served models

`models/registry.json` records the held-out evaluation of the four models the API serves (200k steps, pooled
scenarios). All four are statistically indistinguishable from TWAP (each 95% CI for the paired difference spans 0).

---

## 3. How to read the statistics

* `Δ` and its CI are paired over test windows; "seeds better" counts training seeds whose mean paired difference is
  negative. Seeds are the independent replications; windows of one seed share a trained agent.
* The p-value is a Wilcoxon signed-rank test, which is sensitive to the *typical* window, while the CI is for the
  *mean*, which heavy-tailed windows can move. Where they disagree (for example DQN vs TWAP: 58% of windows better,
  yet the mean CI includes 0) the effect is small and driven by different parts of the distribution; treat such rows as
  "no robust difference".
* About 130 comparisons (16 comparisons × 8 scopes) are reported without multiple-comparison correction. Isolated intervals that just exclude 0
  (for example −0.03) should not be over-read.

## 4. Limitations

* **Synthetic data only.** Conclusions concern this simulator (calibrated, but not real markets) and are not evidence
  about live execution. The regimes the HMM must find are known to exist by construction.
* One order size (100k shares), one horizon (30 minutes), one action set (0/10/25/50% of remaining inventory).
* 100k training steps and a single hyperparameter setting for all variants, fixed after one pilot run and not tuned.
  More training or tuning could change the ranking; the experiments were not designed to find the best possible agent.
* Fill rates are 72–82%: the participation cap makes completion impossible in the thin scenarios, so IS there is
  dominated by the terminal penalty. The penalty (a one-shot liquidation of the remainder at twice the temporary
  impact) is a modelling choice.
* The Alpaca paper-trading path runs in mock mode.

## 5. Reproducibility

- [x] Result files (`results/`) and the exact configuration (`run_config.json`) are committed; `REPORT.md` is generated.
- [x] Seeds control data generation, network initialisation and environment resets.
- [x] Hyperparameters live in code (`src/evaluation/protocol.py`); `config/*.yaml` mirror them and
      `tests/test_config_consistency.py` fails on drift.
- [x] 180+ automated tests, including causality tests (future data cannot change observations or regime beliefs),
      window/protocol tests, an end-to-end experiment smoke test and API hardening tests. CI runs them on every push.
- [x] Trained checkpoints are committed with their evaluation (`models/registry.json`) and can be regenerated with
      `python scripts/train_models.py`.
- [x] Full experiment: `python scripts/run_experiments.py --seeds 42 123 777 2024 31415 7 99 1234 --timesteps 100000 --jobs 16`
      (about 70 minutes on 16 workers).
