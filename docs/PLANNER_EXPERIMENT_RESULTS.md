# Perception-Aware Planner — End-to-End Results (preset 7, 2026-05-29)

## TL;DR

On a fair corner-to-corner SLAM-RMSE test in the LAC simulator, **perception-aware routing does not improve localization — the slope-aware baseline wins.** A traversability penalty fixed a catastrophic failure mode (the perception planner driving into rocks and diverging), but even the mobile, traversability-balanced perception planner is **~1.7× worse than baseline** on RMSE. Extra feature *exposure* did not translate into better SLAM. This points to the two-environment paper framing: **LAC = cautionary/negative result + mechanism analysis; LRO mid-scale = the positive algorithmic demonstration.**

## Setup

- **Environment:** LAC sim, preset 7 (Moon_Map_01, mission subset 6), `perception_aware_agent` navigating by its own SLAM estimate (closed-loop, `USE_GROUND_TRUTH_NAV=False`).
- **Task:** goal-to-goal, corner-to-corner **(−12,−12) → (12,12)** (longest in-map trajectory; rover spawns at ~(−3.5,−6.9) and transits to the start corner first, identically for all variants).
- **Metric:** SLAM RMSE vs ground truth (`positions_rmse_from_poses` in `finalize`), plus **mobility** (steps, stationary %, GT path length) and mean **look-ahead ρ** along the path (feature exposure).
- **Edge cost:** `dist · (1 + α·(slope/θ_ref)^p + β·(1 − ρ_lookahead) + w·trav(cell))`, α=2, θ_ref=10°, p=2, slope-gated at 20°. `β` weights the look-ahead perception reward (ρ the camera sees ahead); `w` weights the traversability (roughness) penalty on the traversed cell.
- **Determinism:** seed 0 (sim deterministic), so baseline and perception runs are reused from the overnight chain; perception+trav was run fresh with identical conditions.

## Results (3-way)

| Variant | β | w | **SLAM RMSE** | Stationary | Steps | GT path | mean look-ahead ρ |
|---|---|---|---|---|---|---|---|
| **baseline** (slope-aware A*) | 0 | 0 | **0.135 m** | 2% | 5,578 | 53.2 m | 0.163 |
| **perception** (look-ahead only) | 2 | 0 | **9.008 m** | **70%** | 19,234 | 115.0 m | 0.344 |
| **perception + traversability** | 2 | 2 | **0.233 m** | 1% | 4,702 | 45.5 m | 0.258 |

Figures: `output/planner_comparison/comparison.png` (overlay + bars); per-run `trajectory_over_dem.png` / `rho_over_trajectory.png` in each `LAC_SIM/output/NavAgent/<run>/`.

## Interpretation

1. **The traversability penalty fixed the catastrophic failure.** Perception-only (β=2, w=0) routed *into* rock-dense terrain: the local ArcPlanner deadlocked (70% stationary, 19k steps to crawl a 37 m planned path) and SLAM **diverged ~9 m** (GT a stuck loop, SLAM estimate sprawling past ±20 m). Adding `w=2` restored mobility to **baseline levels (1% stationary, no divergence, RMSE 0.23 m)** — exactly the intended decoupling: the planner now routes on **smooth ground that overlooks** feature-rich terrain (look-ahead ρ 0.258 > baseline 0.163) instead of driving onto it. Offline this showed as path roughness dropping back to baseline (0.0130 vs 0.0124) while look-ahead ρ stayed ~2× elevated.

2. **But perception routing still does not beat the baseline.** Even mobile and stable, perception+trav's RMSE (**0.233 m**) is **~1.7× worse** than the baseline shortest slope-feasible path (0.135 m), despite seeing ~1.6× more features. **More feature exposure did not improve localization.** The likely reason: feature *count* (validated by Phase 0 at ~0.29) is not the same as *trackable geometry* — features on rocks/at depth discontinuities/shadow edges are harder for VO to track, and the longer route through varied terrain adds drift. The simple, short, smooth path localizes best.

3. **Net:** in the LAC sim, perception-aware routing is **net neutral-to-negative** for SLAM localization. The hypothesis ("route toward feature-rich terrain to localize better") is not supported end-to-end here.

## Confounds resolved along the way (for the record)

- An **earlier** perception run looked like a *win* (RMSE 0.024 vs baseline 0.080) — that was a **stationary-dilution artifact**: a stuck rover sitting still is trivially tracked, pulling RMSE down. Always report mobility alongside RMSE.
- The cost initially used **cell-ρ** (≈ pose-ρ, which Phase 0 showed *anti*-correlates with features); switching to **look-ahead ρ** was correct but did not, by itself, fix traversability.
- Preset 7's first collection was **corrupted** (rover flipped on a rock → ~5k black sky frames). An obstacle-avoiding `phase0_collection_agent` (ArcPlanner + Frontend rock detection + stuck/backup) re-collected it cleanly (10,565 frames). The clean predictor re-validation is **BORDERLINE ~0.29** — i.e. the predictor is real but, as shown above, **not actionable for routing**.

## Caveats

- **n = 1** (start,goal) pair, one preset, deterministic seed. A `β`/`w` sweep and several (start,goal) pairs would harden the claim — but the direction is consistent across all runs (baseline RMSE ~0.13 m; every perception variant is worse).
- LAC terrain is intrinsically smooth with localized rocks; results may differ on terrain with broadly distributed texture (e.g. the real-lunar LRO tile).

## Recommendations

1. **Do not claim a perception-aware localization win in LAC.** The baseline wins; forcing a positive story would be unsupported.
2. **Reframe the paper around the two-environment design:**
   - **LAC (small-scale):** present the **cautionary result + mechanism analysis** — feature-richness ≡ rock-density, so naive perception routing degrades traversability/VO; the traversability term recovers mobility but not a localization win; feature count ≠ trackable geometry. (Strong, honest contribution.)
   - **LRO mid-scale (real lunar DEM, algorithmic, no rocks-as-obstacles / no closed-loop SLAM):** demonstrate the **predicted** benefit (D-optimality / feature exposure along planned paths) where traversability is not a confound.
3. **Keep mobility a first-class metric** in every planner evaluation (RMSE alone misled twice).
4. **If pursuing LAC further:** validate the predictor against *actual VO/SLAM error reduction* (not feature count), and consider a `β`/`w` sweep + multi-(start,goal) campaign via `scripts/run_planner_comparison.sh`.

## Artifacts

- Comparison: `output/planner_comparison/comparison.png`, `output/planner_comparison/results.txt`
- Runs (preset 7, corner-to-corner): baseline `LAC_SIM/output/NavAgent/2026-05-29_00-51-41`, perception `…_01-14-46`, perception+trav `…_09-00-34`
- Clean preset-7 predictor validation: `output/phase0_transect_p7_v2/`
- Tooling: `scripts/run_planner_comparison.sh`, `scripts/compare_runs.py`, `scripts/plot_run_diagnostics.py`
- Configs: `configs/perception_aware_baseline.json` (β0,w0), `configs/perception_aware.json` (β2,w0), `configs/perception_aware_trav.json` (β2,w2)
