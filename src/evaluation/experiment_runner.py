"""
src/evaluation/experiment_runner.py — Full Research Experiment Suite

For every (scenario, seed):

  1. Generate a long regime-switching synthetic series (see scenarios.py).
  2. Chronological split: first 70% train | last 30% test.
  3. Fit the HMM on the train split only.
  4. Evaluate TWAP, VWAP and POV on every non-overlapping test window.
  5. Train each learned model on random train windows and evaluate it on the
     *same* test windows:
        DQN, Regime-Aware DQN, DQN (shuffled-regime control),
        PPO, Regime-Aware PPO, PPO (shuffled-regime control)
  6. Validate the causal HMM against the true latent regime path.

Because every strategy sees identical windows, all comparisons are paired.

Outputs (written to ``results_dir``, never post-processed by hand):
  window_results.csv        one row per (scenario, seed, strategy, window)
  experiment_results.csv    per (scenario, seed, strategy) means over windows
  summary_table.csv         mean / std of IS per strategy
  comparison_table.csv      wide table of mean metrics per strategy
  per_scenario_table.csv    strategy x scenario pivot of mean IS
  paired_comparisons.csv    paired differences with bootstrap CI and Wilcoxon p
  hmm_validation.csv        causal HMM agreement with the true regime
  rq_summary.json           headline deltas
  run_config.json           exact configuration of the run
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation import protocol as P
from src.evaluation.metrics import ResultsAggregator, SingleRunRecord
from src.evaluation.scenarios import SCENARIOS, generate_scenario_data

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS = list(SCENARIOS.keys())
DEFAULT_SEEDS = [42, 123, 777, 2024, 31415]
RESULTS_DIR = "results"

# Comparisons reported in paired_comparisons.csv: (rq, treatment, control)
COMPARISONS: List[Tuple[str, str, str]] = (
    [("RQ1", m, b) for m in (P.NAME_DQN, P.NAME_PPO) for b in P.BASELINE_NAMES]
    + [("RQ1", m, b) for m in (P.NAME_RA_DQN, P.NAME_RA_PPO) for b in P.BASELINE_NAMES]
    + [("RQ2", P.NAME_RA_DQN, P.NAME_DQN), ("RQ2", P.NAME_RA_PPO, P.NAME_PPO)]
    + [("RQ3-control", P.NAME_RA_DQN, P.NAME_SH_DQN), ("RQ3-control", P.NAME_RA_PPO, P.NAME_SH_PPO)]
)
TRANSITION_SCENARIOS = ["regime_transition", "stress", "high_volatility", "liquidity_shock"]


# ── One (scenario, seed) job ──────────────────────────────────────────────────

def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run baselines and learned models for one (scenario, seed). Picklable, self-contained."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # pragma: no cover
        pass

    scenario, seed = job["scenario"], job["seed"]
    timesteps = job["timesteps"]
    t0 = time.time()

    horizon, part = job["horizon"], job["order_participation"]
    df = generate_scenario_data(scenario, n_steps=job["n_bars"], seed=seed)
    split = P.make_split(df)
    starts = P.test_window_starts(split, horizon=horizon, max_windows=job.get("max_windows"))

    frames = [P.evaluate_baseline_windows(split.full, starts, horizon=horizon, order_participation=part)]
    hmm = P.fit_hmm([split.train])
    hmm_row: Dict[str, Any] = {"scenario": scenario, "seed": seed}

    if hmm is None:
        logger.warning("HMM fit failed for %s/%s; skipping learned models", scenario, seed)
    else:
        hmm_row.update(_validate_hmm(hmm, split))
        names = [n for n in P.LEARNED_SPECS
                 if (job["run_ablation"] or "shuffled" not in n) and (job["include_ppo"] or "PPO" not in n)]
        for name in names:
            algo, kind = P.LEARNED_SPECS[name]
            try:
                agent = P.train_agent(algo, kind, [split.train], hmm, seed=seed, timesteps=timesteps,
                                      cut=split.cut, horizon=horizon, order_participation=part)
                env = P.build_env(kind, [split.full], hmm, shuffle_seed=seed + 5000, horizon=horizon,
                                  order_participation=part)
                frames.append(P.evaluate_agent_windows(agent, env, starts, name, seed=seed))
            except Exception as exc:
                logger.error("%s failed for %s/%s: %s", name, scenario, seed, exc)

    win = pd.concat(frames, ignore_index=True)
    win.insert(0, "seed", seed)
    win.insert(0, "scenario", scenario)
    return {"windows": win, "hmm": hmm_row, "seconds": time.time() - t0}


def _validate_hmm(hmm, split: P.SeriesSplit) -> Dict[str, float]:
    """Agreement between the causal filter and the true latent regime on the test region."""
    from sklearn.metrics import adjusted_rand_score
    from src.regimes.inference import CausalRegimeInference

    feats = P.compute_regime_features(split.full)
    cols = hmm.feature_names
    mask = feats[cols].notna().all(axis=1).to_numpy()
    X = feats.loc[mask, cols].to_numpy()
    truth = split.full["true_regime"].to_numpy()[mask]
    inf = CausalRegimeInference(hmm)
    inf.reset_online_state()
    pred = np.array([inf.step_online(x)[0] for x in X])
    test_mask = np.arange(len(pred)) >= (split.cut - int((~mask).sum()))
    return {
        "test_ari": float(adjusted_rand_score(truth[test_mask], pred[test_mask])),
        "test_accuracy": float(np.mean(truth[test_mask] == pred[test_mask])),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def _records_from_windows(win: pd.DataFrame) -> List[SingleRunRecord]:
    recs = []
    for (scen, seed, strat), g in win.groupby(["scenario", "seed", "strategy"], sort=False):
        m = g.mean(numeric_only=True)
        recs.append(SingleRunRecord(
            strategy_name=strat, scenario=scen, seed=int(seed),
            implementation_shortfall_bps=float(m["implementation_shortfall_bps"]),
            execution_cost=float(m["total_execution_cost"]),
            market_impact_cost=float(m["total_impact_cost"]),
            total_transaction_fees=float(m["total_transaction_fees"]),
            completion_rate=float(min(1.0, m["fill_rate"])),
            average_execution_price=float(m["average_execution_price"]),
            arrival_price=float(m["arrival_price"]),
            vwap_slippage_bps=float(m["vwap_slippage_bps"]),
            terminal_penalty=float(m["terminal_penalty"]),
            turnover=float((g["executed_inventory"] * g["average_execution_price"]).mean()),
        ))
    return recs


def paired_comparisons(win: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """
    Paired differences (treatment - control, IS bps; negative = treatment cheaper).

    Two levels are reported for every comparison and scope:
      window : all test windows pooled (captures market-path variance for the
               trained agents; windows share an agent, so treat p-values as
               indicative)
      seed   : per-seed mean difference; the mean/std across independent
               training seeds, plus how many seeds favour the treatment
    """
    piv = win.pivot_table(index=["scenario", "seed", "window"], columns="strategy",
                          values="implementation_shortfall_bps")
    rows = []
    scopes: List[Tuple[str, Optional[List[str]]]] = [("ALL", None)]
    scopes += [(s, [s]) for s in sorted(win["scenario"].unique())]
    present_trans = [s for s in TRANSITION_SCENARIOS if s in set(win["scenario"])]
    if present_trans:
        scopes.append(("TURBULENT", present_trans))

    for rq, treat, ctrl in COMPARISONS:
        if treat not in piv.columns or ctrl not in piv.columns:
            continue
        for scope_name, scope in scopes:
            sub = piv if scope is None else piv[piv.index.get_level_values("scenario").isin(scope)]
            d = (sub[treat] - sub[ctrl]).dropna()
            if d.empty:
                continue
            ws = P.paired_stats(d.to_numpy(), n_boot=n_boot)
            rows.append({"rq": rq, "treatment": treat, "control": ctrl, "scope": scope_name,
                         "level": "window", **ws, "n_seeds": np.nan, "seed_std": np.nan,
                         "seeds_favouring": np.nan})
            per_seed = d.groupby(level="seed").mean()
            rows.append({"rq": rq, "treatment": treat, "control": ctrl, "scope": scope_name,
                         "level": "seed", "n": int(len(per_seed)), "mean": float(per_seed.mean()),
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                         "win_rate": float((per_seed < 0).mean()),
                         "n_seeds": int(len(per_seed)),
                         "seed_std": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                         "seeds_favouring": int((per_seed < 0).sum())})
    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_experiment_suite(
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    train_timesteps: int = 60_000,
    results_dir: str = RESULTS_DIR,
    verbose: int = 0,
    run_ablation: bool = True,
    include_ppo: bool = True,
    n_bars: int = P.N_BARS,
    n_jobs: int = 1,
    max_windows: Optional[int] = None,
    horizon: int = P.HORIZON_STEPS,
    order_participation: Optional[float] = P.ORDER_PARTICIPATION,
) -> ResultsAggregator:
    """
    Execute the full research experiment suite.

    Args:
        scenarios: Scenario names (default: all six).
        seeds: Random seeds (default: five).
        train_timesteps: Training steps per learned model.
        results_dir: Output directory.
        verbose: Unused (kept for API compatibility).
        run_ablation: Include the shuffled-regime control models.
        include_ppo: Include the PPO family (needed for RQ4).
        n_bars: Length of each synthetic series.
        n_jobs: Parallel worker processes over (scenario, seed) jobs.
        max_windows: Optionally cap the number of test windows (smoke tests).
        horizon: Execution horizon in bars (test windows are non-overlapping blocks of this length).
        order_participation: Order size as a fraction of the window's expected volume (None = fixed
            100,000 shares).
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    seeds = seeds or DEFAULT_SEEDS
    os.makedirs(results_dir, exist_ok=True)

    jobs = [dict(scenario=s, seed=sd, timesteps=train_timesteps, n_bars=n_bars,
                 run_ablation=run_ablation, include_ppo=include_ppo, max_windows=max_windows,
                 horizon=horizon, order_participation=order_participation)
            for s in scenarios for sd in seeds]

    config = dict(scenarios=scenarios, seeds=seeds, train_timesteps=train_timesteps, n_bars=n_bars,
                  horizon_steps=horizon, order_participation=order_participation,
                  target_inventory=P.TARGET_INVENTORY, side=P.SIDE,
                  train_ratio=P.TRAIN_RATIO, run_ablation=run_ablation, include_ppo=include_ppo,
                  max_windows=max_windows, n_jobs=n_jobs)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("KAIRO experiment suite: %d jobs (%d scenarios x %d seeds), %s steps/model, %d workers",
                len(jobs), len(scenarios), len(seeds), f"{train_timesteps:,}", n_jobs)

    start = time.time()
    windows: List[pd.DataFrame] = []
    hmm_rows: List[Dict[str, Any]] = []

    def _collect(res: Dict[str, Any], label: str, done: int) -> None:
        windows.append(res["windows"])
        hmm_rows.append(res["hmm"])
        logger.info("[%d/%d] %s done in %.0fs", done, len(jobs), label, res["seconds"])

    if n_jobs <= 1:
        for i, job in enumerate(jobs, 1):
            _collect(_run_job(job), f"{job['scenario']}/seed={job['seed']}", i)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(_run_job, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                _collect(fut.result(), f"{j['scenario']}/seed={j['seed']}", i)

    win = pd.concat(windows, ignore_index=True)
    win.to_csv(os.path.join(results_dir, "window_results.csv"), index=False)
    pd.DataFrame(hmm_rows).to_csv(os.path.join(results_dir, "hmm_validation.csv"), index=False)

    aggregator = ResultsAggregator()
    aggregator.add_many(_records_from_windows(win))
    aggregator.save_csv(os.path.join(results_dir, "experiment_results.csv"))
    try:
        aggregator.save_parquet(os.path.join(results_dir, "experiment_results.parquet"))
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Parquet export skipped: %s", exc)

    aggregator.summary_table().to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)
    aggregator.comparison_table().to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)
    aggregator.per_scenario_table().to_csv(os.path.join(results_dir, "per_scenario_table.csv"), index=False)
    paired_comparisons(win).to_csv(os.path.join(results_dir, "paired_comparisons.csv"), index=False)
    with open(os.path.join(results_dir, "rq_summary.json"), "w") as f:
        json.dump(aggregator.rq_summary(), f, indent=2, default=str)

    logger.info("All runs complete in %.0fs. Results in %s", time.time() - start, os.path.abspath(results_dir))
    return aggregator
