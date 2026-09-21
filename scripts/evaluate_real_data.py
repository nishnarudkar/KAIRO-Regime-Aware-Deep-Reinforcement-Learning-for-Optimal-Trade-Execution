"""
scripts/evaluate_real_data.py — Transfer test on real market bars.

Evaluates TWAP / VWAP / POV and the served (synthetic-trained) RL models on non-overlapping windows of real AAPL
one-minute bars already stored in ``data/raw``. The models never saw real data.

Data caveats (reported, not hidden):
  * The bars come from Alpaca's IEX feed: volume is a small fraction of consolidated volume and many minutes
    have no trade. Bars are placed on a regular 09:30-16:00 grid (price forward-filled, missing volume = 0).
  * There are no quotes, so the simulator applies its default 2 bps spread to every strategy alike.
  * Only a few days exist, so the sample is small and confidence intervals are wide.

Usage:
    python scripts/evaluate_real_data.py [--raw data/raw/aapl_raw_*.parquet] [--out results/real_data]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def regular_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Sparse IEX bars -> complete 1-minute RTH grid (US Eastern), price forward-filled, missing volume = 0."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()
    df = df.between_time("09:30", "15:59")
    days = sorted(set(df.index.normalize()))
    frames = []
    for d in days:
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min")
        day = df.loc[df.index.normalize() == d].reindex(idx)
        if day["close"].notna().sum() < 50:      # skip near-empty days
            continue
        day["close"] = day["close"].ffill().bfill()
        for c in ("open", "high", "low"):
            day[c] = day[c].fillna(day["close"])
        day["volume"] = day["volume"].fillna(0.0)
        frames.append(day)
    g = pd.concat(frames)
    g["price"] = g["close"]
    g["spread"] = g["price"] * 2e-4                 # simulator default spread (2 bps), applied to all strategies
    g["bid"] = g["price"] - g["spread"] / 2
    g["ask"] = g["price"] + g["spread"] / 2
    r = np.log(g["price"]).diff().fillna(0.0)
    g["volatility"] = r.rolling(20, min_periods=5).std().fillna(r.std()).clip(lower=1e-5)
    out = g.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    return out[["timestamp", "open", "high", "low", "close", "price", "bid", "ask", "volume", "spread", "volatility"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default="results/real_data")
    ap.add_argument("--horizon", type=int, default=30)
    args = ap.parse_args()

    from src.agents import registry
    from src.evaluation import protocol as P
    from src.evaluation.experiment_runner import paired_comparisons

    path = args.raw or sorted(glob.glob("data/raw/aapl_raw_*.parquet"))[-1]
    df = regular_grid(pd.read_parquet(path))
    n_days = len(df) // 390
    print(f"{path}: {len(df)} grid bars over {n_days} days; zero-volume minutes: {(df.volume == 0).mean():.1%}; "
          f"mean volume {df.volume.mean():.0f} shares/min")

    horizon, part = args.horizon, P.ORDER_PARTICIPATION
    starts = list(range(P.WARMUP_BARS, len(df) - horizon + 1, horizon))
    # keep windows where the expected volume makes the order meaningful (at least 5 trading minutes of history)
    starts = [s for s in starts if df.volume.iloc[max(0, s - P.HISTORY_BARS):s].sum() > 0]
    print(f"{len(starts)} non-overlapping {horizon}-bar windows")

    frames = [P.evaluate_baseline_windows(df, starts, horizon=horizon, order_participation=part)]
    missing = []
    for policy in registry.RL_POLICIES:
        try:
            lp = registry.load_policy(policy)
        except registry.ModelUnavailable:
            missing.append(policy)
            continue
        env = P.build_env(lp.spec["kind"], [df], lp.hmm, horizon=horizon, order_participation=part)
        frames.append(P.evaluate_agent_windows(lp.agent, env, starts, policy))
    win = pd.concat(frames, ignore_index=True)
    win.insert(0, "seed", 0)
    win.insert(0, "scenario", "aapl_real")

    os.makedirs(args.out, exist_ok=True)
    win.to_csv(os.path.join(args.out, "window_results.csv"), index=False)
    comp = paired_comparisons(win)
    comp.to_csv(os.path.join(args.out, "paired_comparisons.csv"), index=False)
    summ = (win.groupby("strategy").implementation_shortfall_bps.agg(mean="mean", std="std", min="min", max="max",
                                                                     n_runs="count").reset_index().sort_values("mean"))
    summ.to_csv(os.path.join(args.out, "summary_table.csv"), index=False)
    cfg = dict(scenarios=["aapl_real"], seeds=[0], train_timesteps=None, horizon_steps=horizon,
               order_participation=part, side=P.SIDE, n_bars=int(len(df)), source=os.path.basename(path),
               note="Transfer test: models trained on synthetic data only; IEX volume; 2 bps default spread",
               missing_models=missing)
    json.dump(cfg, open(os.path.join(args.out, "run_config.json"), "w"), indent=2)

    fill = win.groupby("strategy").fill_rate.mean().round(3).to_dict()
    print(summ.to_string(index=False))
    print("fill rates:", fill)
    print(comp[(comp.level == "window") & (comp.scope == "ALL")][
        ["rq", "treatment", "control", "n", "mean", "ci_low", "ci_high", "p_value"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
