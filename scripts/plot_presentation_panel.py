#!/usr/bin/env python
"""Single-panel presentation figure: rho_norm + trajectory + sub-goals + detour attempts.

Drops the anchor stars and LC-event x's that clutter the diagnostic plot. Keeps only:
  - rho_norm field (the perception cost the planner used)
  - GT trajectory (black) and SLAM trajectory (deepskyblue)
  - sub-goal waypoints (yellow numbered circles)
  - start (green circle) / end (red square)
  - detour-attempt targets (green diamonds; filled if the detour cleared via verified LC)

Usage:
    python scripts/plot_presentation_panel.py <run_dir> [--dem <path>] [--out <file>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac.planning.perception_map import PerceptionMap
from lac.params import MAP_EXTENT
from lac.util import get_positions_from_poses

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTENT = [-MAP_EXTENT, MAP_EXTENT, -MAP_EXTENT, MAP_EXTENT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dem", type=Path, default=None, help="override DEM .dat path")
    ap.add_argument("--out", type=Path, default=None, help="output PNG path (default: <run_dir>/presentation_panel.png)")
    ap.add_argument("--title", type=str, default=None, help="optional figure title")
    args = ap.parse_args()
    out_path = args.out or (args.run_dir / "presentation_panel.png")

    log = json.loads((args.run_dir / "data_log.json").read_text())
    frames = log["frames"]
    gt_poses = np.array([np.array(f["pose"], dtype=np.float64) for f in frames])
    gt_xy = gt_poses[:, :2, 3]

    slam_path = args.run_dir / "slam_poses.npy"
    slam_xy = (
        get_positions_from_poses(np.load(slam_path))[:, :2]
        if slam_path.exists() else None
    )

    preset = log.get("preset")
    dem_path = args.dem or (REPO_ROOT / "LAC_SIM" / "results" / f"Moon_Map_01_{preset}_rep0.dat")
    pm = PerceptionMap.from_lac_dat(dem_path)

    # v1a-tour: sub-goal sequence from the planner sidecar (omitted gracefully if missing).
    state_path = args.run_dir / "perception_planner_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    goal_sequence = np.asarray(state.get("goal_sequence", []), dtype=np.float64).reshape(-1, 2)
    detour_events = state.get("detour_events", [])

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    im = ax.imshow(
        pm.rho_norm.T, extent=_EXTENT, origin="lower", cmap="magma", aspect="equal"
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho_{\mathrm{norm}}$")

    # Trajectories. GT is bright green (visible against magma's dark regions); SLAM stays sky-blue.
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#00FF7F", lw=1.8, alpha=0.95, label="ground truth")
    if slam_xy is not None:
        ax.plot(slam_xy[:, 0], slam_xy[:, 1], color="deepskyblue", lw=1.8, alpha=0.95, label="SLAM")

    # Start / end markers. Start uses white face with black edge so it doesn't blend into the
    # bright-green GT line; end stays red so it's unmistakable on the magma background.
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], "o", color="white", ms=12, mec="black", mew=1.2, label="start", zorder=6)
    ax.plot(gt_xy[-1, 0], gt_xy[-1, 1], "s", color="red", ms=12, mec="black", mew=1.2, label="end", zorder=6)

    # Detour-attempt targets (kept). Filled = LC-verified success, hollow = cleared via timeout.
    if detour_events:
        plotted_any_success = False
        plotted_any_attempt = False
        for d in detour_events:
            tx, ty = d["target_xy"]
            success = bool(d.get("success", False))
            ax.plot(
                tx, ty, marker="D", ms=12,
                mfc="lime" if success else "none",
                mec="lime", mew=2.0, zorder=7,
            )
            plotted_any_success = plotted_any_success or success
            plotted_any_attempt = plotted_any_attempt or (not success)
        # Legend proxies (single entry each).
        if plotted_any_attempt:
            ax.plot([], [], marker="D", ms=12, mfc="none", mec="lime", mew=2.0,
                    linestyle="", label=f"detour target (n={len(detour_events)})")
        if plotted_any_success:
            ax.plot([], [], marker="D", ms=12, mfc="lime", mec="lime", mew=2.0,
                    linestyle="", label="detour verified")

    # Numbered sub-goal waypoints.
    if len(goal_sequence) > 0:
        ax.scatter(goal_sequence[:, 0], goal_sequence[:, 1], marker="o", s=180,
                   facecolors="yellow", edgecolors="black", linewidths=1.2,
                   zorder=8, label=f"sub-goals (n={len(goal_sequence)})")
        for k, (gx, gy) in enumerate(goal_sequence, start=1):
            ax.annotate(str(k), (gx, gy), color="black", fontsize=10,
                        fontweight="bold", ha="center", va="center", zorder=9)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = args.title or f"{args.run_dir.name}: trajectory over $\\rho_{{\\mathrm{{norm}}}}$"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
