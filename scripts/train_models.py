"""
scripts/train_models.py — Train and save the models served by the API.

Trains DQN, Regime-Aware DQN, PPO and Regime-Aware PPO on random 30-bar windows
pooled from the *train* region of all six scenarios, fits one HMM on the pooled
train features, and stores everything in ``models/`` together with a held-out
evaluation (test region of the same series, paired against TWAP).

Usage:
    python scripts/train_models.py                      # 200k steps per model
    python scripts/train_models.py --timesteps 20000    # quick run
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import datetime
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_models")

DATA_SEED = 1000   # data seeds are disjoint from the experiment seeds


def _series(n_bars: int):
    from src.evaluation.scenarios import SCENARIOS, generate_scenario_data
    from src.evaluation import protocol as P
    return {name: P.make_split(generate_scenario_data(name, n_bars, DATA_SEED + i))
            for i, name in enumerate(SCENARIOS)}


def _train_one(args):
    model_id, timesteps, seed, n_bars, models_dir = args
    os.environ["KAIRO_MODELS_DIR"] = models_dir
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from src.agents import registry
    from src.evaluation import protocol as P

    spec = registry.MODEL_SPECS[model_id]
    splits = _series(n_bars)
    train_dfs = [s.train for s in splits.values()]
    hmm = registry.load_hmm() if spec["regime_aware"] else None
    t0 = time.time()
    agent = P.train_agent(spec["algo"], spec["kind"], train_dfs, hmm, seed=seed,
                          timesteps=timesteps, cut=len(train_dfs[0]))
    agent.save(str(registry.models_dir() / model_id))
    return model_id, time.time() - t0


def _evaluate(n_bars: int):
    """Held-out evaluation on the test windows of every scenario, paired against TWAP."""
    from src.agents import registry
    from src.evaluation import protocol as P

    splits = _series(n_bars)
    hmm = registry.load_hmm()
    rows = []
    loaded = {p: registry.load_policy(p) for p in registry.RL_POLICIES}
    for scen, sp in splits.items():
        starts = P.test_window_starts(sp)
        base = P.evaluate_baseline_windows(sp.full, starts)
        base["scenario"] = scen
        rows.append(base)
        for policy, lp in loaded.items():
            env = P.build_env(lp.spec["kind"], [sp.full], hmm if lp.spec["regime_aware"] else None)
            res = P.evaluate_agent_windows(lp.agent, env, starts, policy)
            res["scenario"] = scen
            rows.append(res)
    win = pd.concat(rows, ignore_index=True)
    piv = win.pivot_table(index=["scenario", "window"], columns="strategy",
                          values="implementation_shortfall_bps")
    out = {}
    for policy in ("TWAP", "VWAP", "POV", *registry.RL_POLICIES):
        sub = win[win.strategy == policy]
        entry = {
            "is_bps_mean": float(sub.implementation_shortfall_bps.mean()),
            "fill_rate_mean": float(sub.fill_rate.mean()),
            "n_windows": int(len(sub)),
        }
        if policy != "TWAP":
            entry["vs_twap_bps"] = P.paired_stats((piv[policy] - piv["TWAP"]).to_numpy())
        out[policy] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description="Train and save KAIRO serving models")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bars", type=int, default=3000)
    parser.add_argument("--models-dir", type=str, default=None)
    args = parser.parse_args()

    if args.models_dir:
        os.environ["KAIRO_MODELS_DIR"] = args.models_dir
    from src.agents import registry
    from src.evaluation import protocol as P

    mdir = registry.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    logger.info("Models directory: %s", mdir)

    splits = _series(args.n_bars)
    hmm = P.fit_hmm([s.train for s in splits.values()], random_state=args.seed)
    if hmm is None:
        raise SystemExit("HMM fit failed")
    hmm.save(str(mdir / registry.HMM_FILENAME))
    logger.info("HMM fitted on pooled train data and saved")

    jobs = [(mid, args.timesteps, args.seed, args.n_bars, str(mdir)) for mid in registry.MODEL_SPECS]
    trained = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for mid, secs in pool.map(_train_one, jobs):
            logger.info("%-11s trained in %.0fs", mid, secs)
            trained[mid] = secs

    registry.clear_cache()
    evaluation = _evaluate(args.n_bars)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write = {
        "version": 1,
        "trained_at": now,
        "protocol": {
            "train": "random 30-bar windows from the train region (first 70%) of all six scenarios",
            "evaluation": "non-overlapping 30-bar windows over the test region (last 30%), paired vs TWAP",
            "data_seed_base": DATA_SEED, "n_bars": args.n_bars, "horizon_steps": P.HORIZON_STEPS,
            "target_inventory": P.TARGET_INVENTORY,
        },
        "models": {
            mid: {"timesteps": args.timesteps, "seed": args.seed, "trained_at": now,
                  "train_seconds": round(trained[mid], 1),
                  "evaluation": evaluation[spec["policy"]]}
            for mid, spec in registry.MODEL_SPECS.items()
        },
        "baselines": {k: evaluation[k] for k in ("TWAP", "VWAP", "POV")},
    }
    registry.write_registry(write)

    logger.info("Held-out evaluation (mean IS bps, paired difference vs TWAP):")
    for name, e in evaluation.items():
        d = e.get("vs_twap_bps")
        extra = f"  vs TWAP {d['mean']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]" if d else ""
        logger.info("  %-18s IS %7.2f  fill %.2f%s", name, e["is_bps_mean"], e["fill_rate_mean"], extra)


if __name__ == "__main__":
    main()
