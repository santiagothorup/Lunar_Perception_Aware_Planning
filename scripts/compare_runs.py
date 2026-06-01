#!/usr/bin/env python
"""Side-by-side comparison of planner-variant runs (baseline vs perception vs perception+trav, ...).

Usage:
    python scripts/compare_runs.py <label:run_dir> [<label:run_dir> ...] [--out <dir>]

Produces <out>/comparison.png:
  - left:  each variant's ground-truth trajectory overlaid on the shared rho_norm map,
  - right: SLAM-RMSE (log scale) and stationary-% bar charts,
and prints a results table (RMSE, steps, GT path length, stationary %, mean look-ahead rho).

Reuses the run-loading / look-ahead / imshow helpers from plot_run_diagnostics.py so there is no
duplicated logic.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import sibling-script helpers (scripts/ is not a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_run_diagnostics import (  # noqa: E402
    load_run, load_lc_events, load_planner_state, _heading_xy, _imshow, REPO_ROOT,
)

from lac.planning.perception_map import PerceptionMap
from lac.planning.perception_aware_planner import LOOKAHEAD_DISTANCES_M
from lac.util import get_positions_from_poses


def run_metrics(run_dir: Path, pm: PerceptionMap) -> dict:
    log, gt_poses, steps, slam = load_run(run_dir)
    gt_xy = gt_poses[:, :2, 3]
    look = float(np.mean([pm.get_lookahead_density(x, y, _heading_xy(p), LOOKAHEAD_DISTANCES_M)
                          for (x, y), p in zip(gt_xy, gt_poses)]))
    plen = float(np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)))
    rp = run_dir / "results.txt"
    rmse = float(np.loadtxt(rp)[0]) if rp.exists() else float("nan")
    if slam is not None and len(slam) > 1:
        sxy = slam[:, :2, 3]
        stationary = float(np.mean(np.linalg.norm(np.diff(sxy, axis=0), axis=1) < 0.005))
        slam_xy = get_positions_from_poses(slam)[:, :2]
    else:
        stationary, slam_xy = float("nan"), None
    # v1c: LC + detour counts for the comparison panel.
    lc_events = load_lc_events(run_dir)
    state = load_planner_state(run_dir)
    return dict(
        rmse=rmse, steps=int(steps.max()), path=plen, stationary=stationary,
        look=look, gt_xy=gt_xy, slam_xy=slam_xy, preset=log.get("preset"),
        n_lc=int(len(lc_events)),
        detour_attempts=int(state.get("detour_attempts", 0)),
        detour_successes=int(state.get("detour_successes", 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label:run_dir entries")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "planner_comparison")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    variants = [(a.split(":", 1)[0], Path(a.split(":", 1)[1])) for a in args.runs]
    # All variants share a preset; build the rho map once from the first run's DEM.
    preset = load_run(variants[0][1])[0].get("preset")
    pm = PerceptionMap.from_lac_dat(REPO_ROOT / "LAC_SIM" / "results" / f"Moon_Map_01_{preset}_rep0.dat")

    m = {label: run_metrics(d, pm) for label, d in variants}

    print(
        f"\n{'variant':<18}{'RMSE(m)':>10}{'steps':>8}{'GTpath(m)':>11}"
        f"{'stationary':>12}{'look-rho':>10}{'LCs':>6}{'detours':>10}"
    )
    for label, _ in variants:
        r = m[label]
        print(
            f"{label:<18}{r['rmse']:>10.3f}{r['steps']:>8}{r['path']:>11.1f}"
            f"{r['stationary']*100:>11.0f}%{r['look']:>10.3f}{r['n_lc']:>6}"
            f"{r['detour_successes']:>5}/{r['detour_attempts']:<4}"
        )

    colors = [f"C{i}" for i in range(len(variants))]
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    _imshow(ax_map, pm.rho_norm, "magma", "rho_norm + anchors + variant ground-truth trajectories")
    # Anchors shared across variants (same preset).
    state0 = load_planner_state(variants[0][1])
    anchors_xy = np.asarray(state0.get("anchors_xy", []), dtype=np.float64).reshape(-1, 2)
    if len(anchors_xy) > 0:
        ax_map.scatter(
            anchors_xy[:, 0], anchors_xy[:, 1], marker="*", s=140, c="cyan",
            edgecolors="black", linewidths=0.6, label="anchors", zorder=5,
        )
    for (label, _), c in zip(variants, colors):
        xy = m[label]["gt_xy"]
        ax_map.plot(xy[:, 0], xy[:, 1], color=c, lw=1.3, alpha=0.9, label=label)
    ax_map.plot(*variants and m[variants[0][0]]["gt_xy"][0], "o", color="lime", ms=9, mec="black", label="start")
    # v1a-tour: overlay numbered sub-goal sequence (shared across variants on a given preset).
    goal_seq = np.asarray(state0.get("goal_sequence", []), dtype=np.float64).reshape(-1, 2)
    if len(goal_seq) > 0:
        ax_map.scatter(goal_seq[:, 0], goal_seq[:, 1], marker="o", s=130, facecolors="yellow",
                       edgecolors="black", linewidths=1.0, zorder=6, label=f"sub-goals (n={len(goal_seq)})")
        for k, (gx, gy) in enumerate(goal_seq, start=1):
            ax_map.annotate(str(k), (gx, gy), color="black", fontsize=9, fontweight="bold",
                            ha="center", va="center", zorder=7)
    ax_map.legend(loc="upper left", fontsize=8)

    labels = [l for l, _ in variants]
    ax_rmse = fig.add_subplot(gs[0, 1])
    bars = ax_rmse.bar(labels, [m[l]["rmse"] for l in labels], color=colors)
    ax_rmse.set_yscale("log")
    ax_rmse.set_ylabel("SLAM RMSE (m, log)")
    ax_rmse.set_title("Localization error (lower = better)")
    for b, l in zip(bars, labels):
        ax_rmse.annotate(f"{m[l]['rmse']:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=8)
    ax_rmse.tick_params(axis="x", rotation=20)

    ax_stat = fig.add_subplot(gs[1, 1])
    ax_stat.bar(labels, [m[l]["stationary"] * 100 for l in labels], color=colors)
    ax_stat.set_ylabel("stationary %")
    ax_stat.set_title("Mobility (lower = better)")
    ax_stat.tick_params(axis="x", rotation=20)

    ax_lc = fig.add_subplot(gs[2, 1])
    bars_lc = ax_lc.bar(labels, [m[l]["n_lc"] for l in labels], color=colors)
    ax_lc.set_ylabel("Loop closures fired")
    ax_lc.set_title("LC count (higher = more re-anchoring)")
    for b, l in zip(bars_lc, labels):
        suf = (
            f"\n({m[l]['detour_successes']}/{m[l]['detour_attempts']} det.)"
            if m[l]["detour_attempts"] else ""
        )
        ax_lc.annotate(f"{m[l]['n_lc']}{suf}",
                       (b.get_x() + b.get_width() / 2, b.get_height()),
                       ha="center", va="bottom", fontsize=7)
    ax_lc.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Planner comparison (preset {preset})")
    fig.tight_layout()
    out = args.out / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
