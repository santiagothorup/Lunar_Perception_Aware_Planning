#!/usr/bin/env python
"""Clean 2-panel figure of the perception cost field for a given preset:
   left  = rho_norm   (normalized feature-density predictor; shadowed cells -> 0)
   right = shadow_mask (1 = in shadow, 0 = sunlit)

No trajectory, no anchors, no LC events, no waypoints -- pure inputs to the planner.

Usage:
    python scripts/plot_perception_field.py --preset 6
    python scripts/plot_perception_field.py --dem LAC_SIM/results/Moon_Map_01_6_rep0.dat
    python scripts/plot_perception_field.py --run LAC_SIM/output/NavAgent/final_runs/kf_revisit_shadow_trav
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

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTENT = [-MAP_EXTENT, MAP_EXTENT, -MAP_EXTENT, MAP_EXTENT]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", type=int, help="MISSIONS_SUBSET index (uses LAC_SIM/results/Moon_Map_01_<preset>_rep0.dat)")
    g.add_argument("--dem", type=Path, help="explicit DEM .dat path")
    g.add_argument("--run", type=Path, help="run dir (reads preset from data_log.json)")
    ap.add_argument("--out", type=Path, default=None, help="output PNG path")
    ap.add_argument("--title", type=str, default=None, help="optional figure suptitle")
    args = ap.parse_args()

    # Resolve DEM path.
    if args.dem is not None:
        dem_path = args.dem
        out_default = Path(f"perception_field_dem.png")
    elif args.preset is not None:
        dem_path = REPO_ROOT / "LAC_SIM" / "results" / f"Moon_Map_01_{args.preset}_rep0.dat"
        out_default = Path(f"perception_field_preset{args.preset}.png")
    else:
        log = json.loads((args.run / "data_log.json").read_text())
        preset = log.get("preset")
        dem_path = REPO_ROOT / "LAC_SIM" / "results" / f"Moon_Map_01_{preset}_rep0.dat"
        out_default = args.run / "perception_field.png"
    out_path = args.out or out_default

    pm = PerceptionMap.from_lac_dat(dem_path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # --- Left: feature-density predictor BEFORE the shadow mask is applied ---
    # rho_norm = roughness_norm * (1 - shadow), so plotting rho_norm zeroes out shadowed cells.
    # The user wanted to see the *roughness-only* component of the predictor without that overlay.
    im0 = axes[0].imshow(
        pm.roughness_norm.T, extent=_EXTENT, origin="lower",
        cmap="magma", aspect="equal", vmin=0, vmax=1,
    )
    axes[0].set_title(r"Roughness (feature-density predictor, no shadow applied)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                 label=r"roughness$_{\mathrm{norm}}$ in [0, 1]")

    # --- Right: shadow_mask ---
    # Use a grayscale-reversed cmap so shadow=1 is dark, sunlit=0 is bright (intuitive).
    im1 = axes[1].imshow(
        pm.shadow_mask.astype(float).T, extent=_EXTENT, origin="lower",
        cmap="gray_r", aspect="equal", vmin=0, vmax=1,
    )
    axes[1].set_title("shadow mask  (1 = in shadow)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="shadow indicator")

    if args.title:
        fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")
    print(f"     mean rho_norm = {float(np.nanmean(pm.rho_norm)):.3f}   "
          f"shadow fraction of map = {float(np.mean(pm.shadow_mask)):.3f}")


if __name__ == "__main__":
    main()
