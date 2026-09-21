"""
scripts/tune_agents.py — Hyperparameter search on TUNING data.

Tuning uses seeds that are disjoint from the experiment / test seeds, and a paired
implementation-shortfall difference vs TWAP on the tuning series' own held-out (last 30%)
windows as the selection metric. Hyperparameters are chosen on the plain (7-dim) agents and
then applied unchanged to the regime-aware and shuffled-regime variants, so no variant is
favoured by the search.

Usage:
    python scripts/tune_agents.py --jobs 28 --timesteps 150000
    python scripts/tune_agents.py --steps-study        # after choosing configs: effect of training length
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import itertools
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tune")

TUNE_SEEDS = [9101, 9102]
TUNE_SCENARIOS = ["normal", "high_volatility", "stress", "regime_transition"]


def search_space():
    cfgs = []
    for lr, net in itertools.product([1e-4, 3e-4, 1e-3], [[64, 64], [128, 128], [256, 256]]):
        cfgs.append(dict(algo="dqn", name=f"dqn lr={lr:g} net={net[0]}", hparams=dict(learning_rate=lr), net_arch=net))
    for lr, ent in itertools.product([1e-4, 3e-4, 1e-3], [0.0, 0.01]):
        cfgs.append(dict(algo="ppo", name=f"ppo lr={lr:g} ent={ent:g} net=128",
                         hparams=dict(learning_rate=lr, ent_coef=ent), net_arch=[128, 128]))
    for net in ([64, 64], [256, 256]):
        cfgs.append(dict(algo="ppo", name=f"ppo lr=0.0003 ent=0 net={net[0]}",
                         hparams=dict(learning_rate=3e-4, ent_coef=0.0), net_arch=net))
    return cfgs


def _run(job):
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from src.evaluation import protocol as P
    from src.evaluation.scenarios import generate_scenario_data

    cfg, scen, seed, steps, horizon = job["cfg"], job["scenario"], job["seed"], job["steps"], job["horizon"]
    df = generate_scenario_data(scen, P.N_BARS if horizon <= 30 else 2 * P.N_BARS, seed)
    sp = P.make_split(df)
    starts = P.test_window_starts(sp, horizon=horizon)
    base = P.evaluate_baseline_windows(sp.full, starts, horizon=horizon, order_participation=P.ORDER_PARTICIPATION)
    twap = base[base.strategy == "TWAP"].implementation_shortfall_bps.to_numpy()
    agent = P.train_agent(cfg["algo"], "plain", [sp.train], None, seed=seed, timesteps=steps, cut=sp.cut,
                          horizon=horizon, order_participation=P.ORDER_PARTICIPATION,
                          hparams=cfg["hparams"], net_arch=cfg["net_arch"])
    env = P.build_env("plain", [sp.full], horizon=horizon, order_participation=P.ORDER_PARTICIPATION)
    res = P.evaluate_agent_windows(agent, env, starts, cfg["name"], seed=seed)
    d = res.implementation_shortfall_bps.to_numpy() - twap
    return dict(config=cfg["name"], algo=cfg["algo"], scenario=scen, seed=seed, steps=steps, horizon=horizon,
                is_bps=float(res.implementation_shortfall_bps.mean()), twap_bps=float(twap.mean()),
                diff_vs_twap=float(d.mean()), fill=float(res.fill_rate.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--out", default="results/tuning")
    ap.add_argument("--steps-study", action="store_true", help="vary training steps for the current best configs")
    ap.add_argument("--best", default=None, help="path to best_configs.json (for --steps-study)")
    ap.add_argument("--only", nargs="+", default=None, help="restrict the search to these config names")
    ap.add_argument("--tag", default=None, help="output file tag (default tuning_results)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.steps_study:
        best = json.load(open(args.best or os.path.join(args.out, "best_configs.json")))
        cfgs = [c for c in search_space() if c["name"] in best.values()]
        steps_list = [50_000, 150_000, 450_000]
        tag = "steps_study"
    else:
        cfgs = [c for c in search_space() if not args.only or c["name"] in args.only]
        steps_list, tag = [args.timesteps], (args.tag or "tuning_results")

    jobs = [dict(cfg=c, scenario=s, seed=sd, steps=st, horizon=args.horizon)
            for c in cfgs for st in steps_list for s in TUNE_SCENARIOS for sd in TUNE_SEEDS]
    log.info("%d runs (%d configs x %s steps x %d series)", len(jobs), len(cfgs), steps_list,
             len(TUNE_SCENARIOS) * len(TUNE_SEEDS))
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 10 == 0:
                log.info("%d/%d done", i, len(jobs))
                pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{tag}.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{tag}.csv"), index=False)

    key = ["config", "steps"] if args.steps_study else ["config"]
    summ = (df.groupby(key).agg(algo=("algo", "first"), diff_vs_twap=("diff_vs_twap", "mean"),
                                sd_across_series=("diff_vs_twap", "std"), is_bps=("is_bps", "mean"),
                                fill=("fill", "mean"), n=("seed", "count")).reset_index()
            .sort_values(["algo", "diff_vs_twap"]))
    summ.to_csv(os.path.join(args.out, f"{tag}_summary.csv"), index=False)
    print(summ.to_string(index=False))
    if not args.steps_study and not args.only:
        best = {a: g.sort_values("diff_vs_twap").iloc[0]["config"] for a, g in summ.groupby("algo")}
        json.dump(best, open(os.path.join(args.out, "best_configs.json"), "w"), indent=2)
        print("best:", best)


if __name__ == "__main__":
    main()
