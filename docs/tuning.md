# Hyperparameter Tuning

`scripts/tune_agents.py` selects hyperparameters on data the experiments and served models never see: tuning seeds
`9101, 9102` (experiment seeds are `42, 123, 777, 2024, 31415, ...`) on four of the six scenarios (`normal`,
`high_volatility`, `stress`, `regime_transition`; `low_liquidity` and `liquidity_shock` are held out entirely).
Selection metric: mean paired implementation-shortfall difference vs. TWAP, on the plain (7-dim) agent only — the
same hyperparameters are then reused unchanged for the regime-aware and shuffled-regime variants, so the search
cannot favour one variant.

## Search space

DQN: learning rate ∈ {1e-4, 3e-4, 1e-3} × network ∈ {[64,64], [128,128], [256,256]} (9 configs).
PPO: learning rate ∈ {1e-4, 3e-4, 1e-3} × entropy coefficient ∈ {0, 0.01} (6 configs), plus 2 network-width configs
at the default learning rate. All at 150,000 training steps, original fraction-based action space (this search ran
before the TWAP-relative action space was introduced — see below).

## Round 1 — original (fraction-based) action space

Full results: `results/tuning/tuning_results_summary.csv`.

| Best per algorithm | Δ vs TWAP (bps) | sd across series |
|---|---|---|
| DQN lr=3e-4 net=128 | +1.74 | 2.38 |
| PPO lr=1e-4 net=128 | +1.69 | 2.99 |

Every one of the 15 configurations tried was **costlier than TWAP** (+1.7 to +11.6 bps). Because the fraction action
set (0/10/25/50% of remaining inventory) cannot express a uniform TWAP schedule, no configuration could match the
baseline regardless of hyperparameters — the ceiling was in the action space, not the learning.

`dqn lr=3e-4 net=128` and `ppo lr=3e-4 ent=0 net=128` were selected (PPO's nominal best, lr=1e-4, was within noise
of lr=3e-4 and less stable at longer training, so the mid learning rate was kept for consistency with DQN).

## Round 2 — validating the TWAP-relative action space

With the action space changed to multiples of the TWAP slice (see `research_audit.md` §1) and re-evaluated on the
same tuning data (`results/tuning/twap_mode_validation_summary.csv`):

| Config | Δ vs TWAP (bps) | sd |
|---|---|---|
| dqn lr=3e-4 net=128 | +1.91 | 2.62 |
| ppo lr=3e-4 ent=0 net=128 | **+0.51** | 2.20 |
| ppo lr=1e-4 ent=0 net=128 | +11.99 | 30.96 |

The action-space change alone did not help DQN (as expected — DQN was not yet penalised by an unreachable TWAP
baseline in the old space either, since it's a value-based method that can already choose action `0` at every step
and approximate low participation). It roughly halved PPO's gap at lr=3e-4 and made lr=1e-4 unstable, confirming
lr=3e-4 as the better choice for PPO.

## Round 3 — validating the drift-free training reward

Same configs, with `drift_free_reward=True` added (`results/tuning/driftfree_validation_summary.csv`):

| Config | Δ vs TWAP (bps) | sd |
|---|---|---|
| dqn lr=3e-4 net=128 | **−0.89** | 1.49 |
| dqn lr=1e-3 net=128 | −0.71 | 1.58 |
| ppo lr=3e-4 ent=0 net=128 | **−0.83** | 0.80 |
| ppo lr=1e-3 ent=0 net=128 | −0.16 | 1.19 |

Both algorithms cross from costlier-than-TWAP to cheaper-than-TWAP on the tuning data, and the run-to-run spread
(`sd`) drops sharply — the reward fix does what §1 of the audit claims: it removes noise from the training signal
rather than changing what is being optimised (evaluation metrics are unaffected; see
`tests/test_design_v2.py::test_drift_free_flag_does_not_change_reported_shortfall`).

**Final choice**, used by `scripts/train_models.py` and `scripts/run_experiments.py`: DQN lr=3e-4 net=[128,128];
PPO lr=3e-4 ent_coef=0 net=[128,128]; both with `action_mode="twap_multiple"` and `drift_free_reward=True`.

## What tuning does not show

The tuning-data numbers above are optimistic relative to the held-out experiment seeds (they are, after all, the
best of a search): compare to the main suite's actual result (DQN +0.50 bps, PPO −0.14 bps vs. TWAP, both CI
including 0) in `research_audit.md` §3. The gap between "-0.89 on 2 tuning seeds" and "+0.50 on 5 held-out seeds"
is a reminder that a handful of seeds is not enough to lock in a hyperparameter choice with confidence — it is
adequate for picking a reasonable configuration, not for claiming a precise effect size.

## Training-length study

`scripts/tune_agents.py --steps-study` (50k / 150k / 450k steps) was scripted but not run to completion in this
round; `results/tuning/` and this document should be updated if it is run later.

## Reproducing

```bash
python scripts/tune_agents.py --jobs 30 --timesteps 150000                                    # round 1
python scripts/tune_agents.py --jobs 15 --timesteps 150000 --tag twap_mode_validation \
    --only "dqn lr=0.0003 net=128" "ppo lr=0.0003 ent=0 net=128" "ppo lr=0.0001 ent=0 net=128"  # round 2
python scripts/tune_agents.py --jobs 32 --timesteps 150000 --tag driftfree_validation \
    --only "dqn lr=0.0003 net=128" "dqn lr=0.001 net=128" \
           "ppo lr=0.0003 ent=0 net=128" "ppo lr=0.001 ent=0 net=128"                            # round 3
```
