#!/usr/bin/env python
"""Post-run diagnostics for a perception-aware (or baseline) agent mission.

Given a run output directory (the `output/<Agent>/<timestamp>/` produced by NavAgent /
PerceptionAwareAgent), produces two figures:

  1. trajectory_over_dem.png  -- the ground-truth + SLAM trajectories overlaid on the preset's
     DEM-derived fields (elevation, roughness, shadow, normalized rho), like the phase-0 overlays.
  2. rho_over_trajectory.png  -- the LOOK-AHEAD feature density the rover actually saw along the
     ground-truth path (what the perception-aware cost optimizes), plus the path colored by it, and
     a mobility readout (steps, path length, stationary fraction) that exposes stuck behavior.

Reads `data_log.json` (preset + per-step GT poses) and `slam_poses.npy`; the DEM is taken from
`LAC_SIM/results/Moon_Map_01_<preset>_rep0.dat`. Self-contained (numpy + matplotlib + PerceptionMap).

Usage:
    python scripts/plot_run_diagnostics.py <run_dir> [--dem <path>] [--out <dir>]
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
from lac.planning.perception_aware_planner import LOOKAHEAD_DISTANCES_M
from lac.params import MAP_EXTENT
from lac.util import get_positions_from_poses

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTENT = [-MAP_EXTENT, MAP_EXTENT, -MAP_EXTENT, MAP_EXTENT]


def _heading_xy(pose: np.ndarray) -> tuple[float, float]:
    """World-frame XY heading (rover +X forward), matching phase0_validation.lookahead_xy."""
    h = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    return float(h[0]), float(h[1])


def _imshow(ax, field: np.ndarray, cmap: str, title: str):
    """imshow a (MAP_SIZE, MAP_SIZE) (x, y)-oriented field in world coords (x right, y up)."""
    im = ax.imshow(field.T, extent=_EXTENT, origin="lower", cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _overlay_trajectories(ax, gt_xy, slam_xy, goal_sequence=None):
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", lw=1.0, alpha=0.85, label="ground truth")
    if slam_xy is not None:
        ax.plot(slam_xy[:, 0], slam_xy[:, 1], color="deepskyblue", lw=1.0, alpha=0.85, label="SLAM")
    ax.plot(gt_xy[0, 0], gt_xy[0, 1], "o", color="lime", ms=8, mec="black", label="start")
    ax.plot(gt_xy[-1, 0], gt_xy[-1, 1], "s", color="red", ms=8, mec="black", label="end")
    # v1a-tour: numbered sub-goal waypoints overlaid so the planned mission shape is visible.
    if goal_sequence is not None and len(goal_sequence) > 0:
        gs = np.asarray(goal_sequence, dtype=float).reshape(-1, 2)
        ax.scatter(gs[:, 0], gs[:, 1], marker="o", s=110, facecolors="yellow",
                   edgecolors="black", linewidths=1.0, zorder=6, label=f"sub-goals (n={len(gs)})")
        for k, (gx, gy) in enumerate(gs, start=1):
            ax.annotate(str(k), (gx, gy), color="black", fontsize=8, fontweight="bold",
                        ha="center", va="center", zorder=7)


def load_run(run_dir: Path):
    log = json.loads((run_dir / "data_log.json").read_text())
    frames = log["frames"]
    poses = np.array([np.array(f["pose"], dtype=np.float64) for f in frames])  # (N,4,4) GT
    steps = np.array([int(f["step"]) for f in frames])
    slam_path = run_dir / "slam_poses.npy"
    slam = np.load(slam_path) if slam_path.exists() else None
    return log, poses, steps, slam


def load_lc_events(run_dir: Path) -> np.ndarray:
    """Return (K, 2) array of (anchor_kf_idx, current_pose_idx) loop-closure events, or empty."""
    p = run_dir / "backend_state.npz"
    if not p.exists():
        return np.empty((0, 2), dtype=int)
    z = np.load(p, allow_pickle=True)
    lcs = z.get("loop_closures") if hasattr(z, "get") else z["loop_closures"]
    arr = np.asarray(lcs)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    return arr.reshape(-1, 2).astype(int)


def load_planner_state(run_dir: Path) -> dict:
    """Return the perception planner sidecar (anchors, detour events). {} if missing."""
    p = run_dir / "perception_planner_state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dem", type=Path, default=None, help="override DEM .dat path")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: run_dir)")
    args = ap.parse_args()
    out_dir = args.out or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log, gt_poses, steps, slam = load_run(args.run_dir)
    preset = log.get("preset")
    dem_path = args.dem or (REPO_ROOT / "LAC_SIM" / "results" / f"Moon_Map_01_{preset}_rep0.dat")
    pm = PerceptionMap.from_lac_dat(dem_path)

    gt_xy = gt_poses[:, :2, 3]
    slam_xy = get_positions_from_poses(slam)[:, :2] if slam is not None else None

    # Look-ahead rho the rover saw at each GT pose (what the perception cost optimizes), and the
    # pose-rho underfoot for contrast (Phase 0: pose-rho anti-correlates, look-ahead correlates).
    look_rho = np.array([pm.get_lookahead_density(x, y, _heading_xy(p), LOOKAHEAD_DISTANCES_M)
                         for (x, y), p in zip(gt_xy, gt_poses)])
    pose_rho = np.array([pm.get_feature_density(x, y) for x, y in gt_xy])
    dist_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1))])

    # v1c diagnostics: anchors, loop-closure events, detour attempts.
    planner_state = load_planner_state(args.run_dir)
    anchors_xy = np.asarray(planner_state.get("anchors_xy", []), dtype=np.float64).reshape(-1, 2)
    # v1a-tour: sub-goal sequence (may be empty for pre-v1a-tour runs -> overlay just skipped).
    goal_sequence = np.asarray(planner_state.get("goal_sequence", []), dtype=np.float64).reshape(-1, 2)
    if len(anchors_xy) == 0:  # fall back to recomputing if the run predates the sidecar.
        from lac.planning.dem import DEM
        anchors_xy = pm.compute_anchors(DEM.from_lac_dat(dem_path))
    detour_events = planner_state.get("detour_events", [])
    n_detour_att = int(planner_state.get("detour_attempts", len(detour_events)))
    n_detour_ok = int(planner_state.get("detour_successes", sum(1 for d in detour_events if d.get("success"))))
    lc_events = load_lc_events(args.run_dir)
    # Map each LC event's current_pose_idx -> a (x, y) on the GT trajectory.
    # backend.pose_idx counts keyframes/poses; align via min(idx, len(gt_xy)-1).
    if len(lc_events) > 0:
        lc_pose_idx = np.clip(lc_events[:, 1], 0, len(gt_xy) - 1)
        lc_xy = gt_xy[lc_pose_idx]
    else:
        lc_xy = np.empty((0, 2))

    # Mobility readout (exposes stuck/stationary behavior).
    if slam is not None and len(slam) > 1:
        sxy = slam[:, :2, 3]
        stationary = float(np.mean(np.linalg.norm(np.diff(sxy, axis=0), axis=1) < 0.005))
    else:
        stationary = float("nan")
    readout = (f"steps: {int(steps.max())}   GT path: {dist_cum[-1]:.1f} m   "
               f"mean look-ahead rho: {look_rho.mean():.3f}   stationary: {stationary*100:.0f}%   "
               f"LCs: {len(lc_events)}   detours: {n_detour_ok}/{n_detour_att} ok")

    # ---- Figure 1: trajectories over DEM fields (+ anchors, LC events on the rho panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _imshow(axes[0, 0], pm.z, "terrain", "Elevation")
    _imshow(axes[0, 1], pm.roughness, "viridis", "Roughness (5x5 std)")
    _imshow(axes[1, 0], pm.shadow_mask.astype(float), "gray_r", "Shadow (1 = in shadow)")
    _imshow(axes[1, 1], pm.rho_norm, "magma", "rho_norm + anchors + LC events")
    for ax in axes.flat:
        _overlay_trajectories(ax, gt_xy, slam_xy, goal_sequence=goal_sequence)
    # Anchor + LC overlays go on the rho panel (1, 1).
    if len(anchors_xy) > 0:
        axes[1, 1].scatter(
            anchors_xy[:, 0], anchors_xy[:, 1],
            marker="*", s=140, c="cyan", edgecolors="black", linewidths=0.6, label="anchors", zorder=5,
        )
    if len(lc_xy) > 0:
        axes[1, 1].scatter(
            lc_xy[:, 0], lc_xy[:, 1],
            marker="x", s=60, c="red", linewidths=1.4, label=f"LC events ({len(lc_xy)})", zorder=6,
        )
    # Detour targets (open diamonds; filled if the detour succeeded).
    for d in detour_events:
        tx, ty = d["target_xy"]
        axes[1, 1].plot(
            tx, ty, marker="D", ms=10,
            mfc="lime" if d.get("success") else "none",
            mec="lime", mew=1.5, zorder=5,
        )
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[1, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{args.run_dir.name}: trajectories over preset-{preset} DEM\n{readout}")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_over_dem.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: rho the rover saw along the trajectory ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].plot(dist_cum, look_rho, color="C0", lw=1.2, label="look-ahead rho (camera view)")
    axes[0].plot(dist_cum, pose_rho, color="C7", lw=0.8, alpha=0.6, label="pose rho (underfoot)")
    axes[0].set_xlabel("distance along GT path (m)")
    axes[0].set_ylabel("rho (normalized)")
    axes[0].set_title("Feature density along trajectory")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    _imshow(axes[1], pm.rho_norm, "magma", "GT path colored by look-ahead rho")
    sc = axes[1].scatter(gt_xy[:, 0], gt_xy[:, 1], c=look_rho, cmap="viridis", s=6, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="look-ahead rho")
    axes[1].plot(gt_xy[0, 0], gt_xy[0, 1], "o", color="lime", ms=8, mec="black")
    axes[1].plot(gt_xy[-1, 0], gt_xy[-1, 1], "s", color="red", ms=8, mec="black")
    fig.suptitle(f"{args.run_dir.name}: rho exposure along trajectory\n{readout}")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_over_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote trajectory_over_dem.png + rho_over_trajectory.png to {out_dir}")
    print(f"     {readout}")
    print(
        f"     anchors: {len(anchors_xy)}   loop_closures: {len(lc_events)}   "
        f"detour_attempts: {n_detour_att}   detour_successes: {n_detour_ok}"
    )


if __name__ == "__main__":
    main()
