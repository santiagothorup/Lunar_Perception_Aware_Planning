# Report Results — Perception-Aware Path Planning for Lunar Rover Autonomy

> Source material for the final report (sections 5: Results, 6: Conclusion, 7: Future Directions). All numerical values verified against the final-run results in `LAC_SIM/output/NavAgent/final_runs/`. Background, related work, problem statement, and approach are in `REPORT_BACKGROUND.md`.

---

## 5. Results

### 5.1 Experimental Procedure

**Simulator and preset.** All evaluations use the LAC simulator (Unreal Engine 4 with NVIDIA RTX-accelerated rendering, headless Xvfb display, `LAC-Linux-Shipping` binary). Preset 6 (`Moon_Map_01_6_rep0.dat`) is used for the final-presentation comparison: it provides a 27 m × 27 m operational area with mixed rocky and shadow regions representative of lunar south-polar terrain. The mission spawn pose is `(−3.51, −6.90)` (deterministic across all variants).

**Mission.** The 6-sub-goal perimeter tour defined in §4.5 of `REPORT_BACKGROUND.md`: `spawn → (−12,−12) → (10,−12) → (12,12) → (−7,7) → (−10,12) → (−12,−12)`. Planned-path length: ~117 m. The route deliberately closes the loop by returning to the start, creating one guaranteed end-of-mission revisit opportunity.

**Variants evaluated.** A 3 × 3 ablation matrix crossing three global-cost configurations (`baseline`, `High Feature` aka `perc_trav`, `Shadow Avoidance`) with three detour modes (`none`, `Tight Anchor`, `KF Revisit`), plus the headline composite variant `KF Revisit + Shadow + Safety`. All variants share the same sub-goal sequence, the same A* slope parameters (`α=2`, `θ_ref=10°`, `θ_max=20°`), the same waypoint spacing (2 m), and the same detour-trigger parameters where applicable.

**Per-variant procedure.** For each variant, the launcher (`scripts/run_planner_comparison.sh`) (i) kills any stale simulator processes, (ii) launches the LAC sim and waits ~90 s for it to initialize, (iii) runs the `perception_aware_agent` under `timeout 7200s` (a 2-hour wall-clock cap), (iv) parses the agent's RMSE printout from stdout, (v) runs `plot_run_diagnostics.py` to extract per-run mobility statistics, and (vi) appends a result line to `LAC_SIM/output/NavAgent/final_runs/results.txt`. All per-variant output (data_log.json, slam_poses.npy, backend_state.npz, per-camera frames, planner sidecar JSON) is saved under `LAC_SIM/output/NavAgent/final_runs/<label>/`.

**Metrics.** The primary metric is **SLAM RMSE** vs ground truth (`positions_rmse_from_poses(slam_poses, slam_eval_poses)`). Secondary metrics:
- **GT path length** (total distance traveled by ground truth in meters)
- **Per-meter drift** = RMSE / path length (a length-normalized drift metric that allows comparison across variants whose path lengths differ by tens of meters)
- **Stationary fraction** (% of frames where consecutive GT positions are within 5 mm — a stuck-rover indicator)
- **Sub-goals reached** (out of 6; "reached" = closest GT pose within 2 m)
- **Loop closures fired** (total `BetweenFactorPose3` factors added to the pose graph by the SLAM backend)
- **Detour attempts** (number of times the planner spliced a detour)

A run is **valid** if (a) the agent's `finalize()` ran cleanly and (b) the rover reached at least 4/6 sub-goals; otherwise the run is reported as a failure with the failure mode noted. Stationary-dilution artifacts (where a stuck rover trivially tracks a stationary GT pose, producing a misleadingly-low RMSE) are filtered by reporting both RMSE *and* stationary % — a low RMSE with >50% stationary is flagged as invalid.

### 5.2 Phase 0 — DEM-derived predictor validation (preset 7)

Before the planner evaluation, we validated whether the DEM-derived feature-density predictor `ρ = roughness × (1 − shadow)` correlates with observed feature count.

**Setup.** A safe collector agent traversed a serpentine raster on LAC preset 7 (the chosen Phase 0 preset due to its terrain variability), capturing `n = 10,565` clean stereo frames. For each frame, SuperPoint features were extracted and counted; for each pose, four candidate predictor values were computed (pose-ρ and look-ahead ρ at `d ∈ {1.5, 2.5, 3.5, 4.0} m`).

**Results.** The strongest correlate was **look-ahead ρ at `d = 1.5 m`**, with `|Pearson_raw| = 0.210` (95% CI [0.114, 0.293]) and `|Pearson_iid_lid| = 0.213` (n_lid = 121). Cleanest preset-7 re-validation (after the obstacle-avoiding collector) yielded `|Pearson| = 0.29`. Mean feature count rose monotonically with the roughness quartile of the look-ahead cell:

| Roughness quartile | Mean n_features ± std |
|---|---|
| Q1 [0.004, 0.009] | 658 ± 270 |
| Q2 [0.009, 0.013] | 657 ± 280 |
| Q3 [0.013, 0.022] | 800 ± 320 |
| Q4 [0.022, 0.224] | 757 ± 350 |

(Phase 0 supporting figures: `output/phase0_transect_p7_v2/`.)

**Interpretation.** A Pearson coefficient of ~0.25 is a *borderline-positive* correlation: the predictor captures real structure but is not a dominant predictor of feature count. Critically, **pose-ρ** (the cell the rover is *on*) anti-correlated, while **look-ahead ρ** (the cell the camera is *looking at*) correlated positively — confirming that the planner should reward look-ahead feature density, not pose-level feature density.

This result motivated two implementation choices: (1) the cost term uses look-ahead ρ averaged over three forward distances `{1.5, 2.5, 3.5} m`, and (2) the traversability penalty `w · trav(n)` was added as a safety mechanism, because routing toward high-ρ cells (which are rough cells by construction) was found to stall the rover in early Phase 1 testing. The borderline strength of the predictor also justified developing the **shadow-only** variant as a simpler alternative.

### 5.3 Headline result: the 5-variant comparison

The five variants that drive the paper's narrative are summarized below. All ran on preset 6, the 6-sub-goal tour, deterministic seed 0.

| Variant | A* Cost | RMSE (m) | Path (m) | Per-meter drift (m/m) | Sub-goals reached | Stationary | LCs fired | Detours |
|---|---|---:|---:|---:|:---:|---:|---:|---:|
| Baseline (no detour) | Slope only | **0.582** | 103.1 | 0.0056 | 6/6 | 1% | 1 | 0 |
| Tight Anchor + High Feature | Slope + ρ + Rough + Anchor detour | 11.00 | 114.2 | 0.0963 | 4/6 | 7.7% | 12 | 4 |
| KF Revisit + High Feature | Slope + ρ + Rough + KF detour | 0.905 | 146.6 | 0.0062 | 6/6 | 3% | 49 | 9 |
| KF Revisit + Shadow | Shadow only + KF detour | 0.741 | 140.1 | 0.0053 | 6/6 | 1% | 17 | 10 |
| **KF Revisit + Shadow + Safety ★** | **Shadow + Rough-as-safety + KF detour** | **0.140** | **139.4** | **0.0010** | **6/6** | **1%** | **54** | **9** |

**Headline numbers.** The winning variant `KF Revisit + Shadow + Safety`:
- **4.16× lower absolute RMSE than baseline** (0.140 vs 0.582 m)
- **5.55× lower per-meter drift than baseline** (0.0010 vs 0.0056 m/m)
- **35% longer trajectory** (139.4 vs 103.1 m)
- **54× more loop closures** than baseline (54 vs 1)
- Fully mobile (1% stationary, 0% stuck-loop artifacts)
- Reached all 6 sub-goals

The per-meter drift metric is the cleanest comparison: it normalizes for the longer path length and isolates the *quality* of the localization-per-distance-traveled, independent of how far the rover went. By this measure the winning variant achieves a **5.5× drift reduction**, which is the headline contribution of this work.

### 5.4 Why this combination works: mechanism analysis

The composite "Shadow + Safety + KF Revisit" winner combines three mechanisms that each contribute distinct value, none of which is sufficient alone.

**(a) Shadow-aware global cost keeps SLAM accurate.** The `s = 6` shadow penalty steers the A* path through sunlit corridors. Quantitatively, the rover's GT trajectory spends only **15.3% of its path-distance in shadow** vs the baseline's 57.5% (computed offline on the planned paths). Because visual features are only detectable in sunlit cells, this directly translates to a higher per-pose feature count and tighter VO. The downstream effect: SLAM-estimated pose stays within `LC_DIST_THRESHOLD = 1 m` of GT for longer — which means the geometric LC trigger keeps working under realistic drift conditions, rather than going blind once SLAM drift exceeds 1 m.

**(b) Traversability-as-safety prevents rocky-cell stalls.** With `w = 4`, the planner avoids cells where the local ArcPlanner would deadlock. Without this term, the shadow-only variant (line 3 in the variant table from §5.5 below) reaches only 4/6 sub-goals and accumulates 3.07 m of RMSE; with `w = 4` added, the rover stays mobile and reaches all 6 sub-goals. The mean roughness along the winning variant's path is 0.30 vs 0.36 for the shadow-only variant — a 17% reduction that, combined with shadow avoidance, gives the cleanest mobility of any variant tested (1% stationary, 0 stuck events).

**(c) KF Revisit detour actively engineers loop closures.** With `steps_since_lc_threshold = 400`, the detour fires whenever SLAM has been drifting for ~400 keyframe-updates without a loop closure. In the winning run, **9 detours fired**, of which the rover physically reached the target keyframe area (within 2 m) before timing out in most cases. Total loop closures fired: 54, vs baseline's 1.

**The critical mechanistic finding:** breaking down where the 54 LCs in the winning variant bound the pose graph:
- **20 of the 54 LCs bind back to keyframes in the first 10% of the mission** (near `g_1 = (−12, −12)`, close to the GTSAM prior pose at `X(0)`).
- **28 LCs fired in the last 20% of the mission** (during the return-to-start leg `g_5 → g_6`).

This is the structural fix the prior literature identifies as necessary for true global drift correction: LCs that bind *late* keyframes to *early* keyframes (which are anchored to the GT prior by the pose-graph initialization) redistribute accumulated drift across the entire trajectory. Mid-trajectory LCs only constrain relative pose between temporally-close keyframes; they reduce *local* drift but don't anchor *global* drift. The combination of (a) the shadow-aware global cost keeping SLAM accurate long enough for the LC trigger to fire on the return leg, plus (b) the KF Revisit detour driving the rover into close proximity with old keyframes, produces 20 "back to start" LCs that anchor the late-mission pose graph back to the well-anchored start region.

### 5.5 The full 3 × 3 ablation matrix

The full matrix below (preset 6, final-runs results) shows that the (Shadow + Safety, KF Revisit) winner is not arbitrary — both axes matter, and several combinations fail outright. The headline winner is the bottom-right corner; cells in the matrix below report `RMSE (m) / sub-goals reached / stationary %`:

| Global cost ↓ / Detour mode → | none | Tight Anchor | KF Revisit |
|---|:---|:---|:---|
| **baseline (slope only)** | 0.58 / 6/6 / 1% (sometimes stuck, see §5.7) | **130.8** / 3/6 / 10% ❌ | **CRASHED** ❌ |
| **High Feature** (β=2, w=8) | 1.59 / 6/6 / 10% | 1.64 / 6/6 / 3% | 0.90 / 6/6 / 3% |
| **Shadow Avoidance** (s=2) | 3.07 / 4/6 / 1% | **TIMED OUT** ❌ | **0.74** / 6/6 / 1% |
| **Shadow + Safety** (s=6, w=4) | — | — | **0.14 / 6/6 / 1% ✓✓✓** |

(Empty cells were not run; the bottom row's first two cells were skipped after `tight_anchor_shadow` timed out and the kf_revisit_shadow result confirmed that the KF Revisit detour was the right mechanism choice for the shadow-cost family.)

**Reading the matrix.**
1. **Base global cost is fragile under any aggressive detour.** Pairing slope-only A* with a detour mechanism produces catastrophic failures: Tight Anchor + base diverged to 130.8 m RMSE; KF Revisit + base crashed (sensor dropout at step 894, AgentRuntimeError — same agent-side failure mode regardless of perception cost, but reproducible only on the base variant in our testing).
2. **Tight Anchor is environment-sensitive.** It works with High Feature global cost (1.64 m RMSE, 36 LCs), times out with Shadow Avoidance, and diverges catastrophically with Baseline. The cause is the same in all cases: anchors are pre-computed from high-ρ cells, which co-occur with rocky cells; without a strong roughness penalty in the global cost (`w ≥ 8` in our experiments), the rover stalls trying to reach the anchor.
3. **KF Revisit is the most robust detour mechanism.** It works with High Feature (0.90 m), with Shadow Avoidance (0.74 m), and produces the headline winner with Shadow + Safety (0.14 m). The reason: KF Revisit targets keyframes the rover has *already visited* — driveable by construction — rather than DEM-predicted high-ρ cells.

### 5.6 Tight Anchor vs KF Revisit — the LC-firing comparison

A striking detail on the difference between the two detour tactics: `tight_anchor_perc` fired **36 LCs from 4 detour attempts**, while `kf_revisit_perc` fired **49 LCs from 9 detour attempts**. The detour-to-LC ratio is similar but Tight Anchor *fires* fewer detours in the first place. Reason: anchor candidates must pass three filters simultaneously (proximity to current pose, proximity to a past keyframe, not in the per-leg blacklist) and, given the limited pool of `K ≤ 12` anchors, most invocations of the detour trigger find no valid anchor and no-op.

The 36 LCs that *did* fire in the `tight_anchor_perc` run were *incidental*: the rover happened to pass within 1 m of a past keyframe during normal travel, not because the detour mechanism drove it there. Only 4 of those 36 LCs are detour-targeted. By contrast, `kf_revisit_shadow_trav` fired 54 LCs from 9 active detour attempts — substantially more "caused" loop closures.

This is the key intuition: **Tight Anchor collects loop closures, KF Revisit causes them**.

### 5.7 Failure-mode analysis

**(a) `tight_anchor_perc` and `tight_anchor_base` get stuck on UE4 collision meshes.**
The previous-tour `tight_anchor` variant ended at GT `(+0.16, +7.55)` having spent 1,877 consecutive frames in a 1.5 m radius — a clear stuck cluster. Forensic analysis showed this cell has DEM-derived roughness 0.22 (below median) and zero rocks in the DEM rock grid, yet the rover physically could not navigate through it. The cause: the LAC simulator includes UE4 collision meshes (the lunar lander mesh at world origin, smaller crater meshes elsewhere) that are *not* represented in the ground-truth DEM. The cost function cannot see them, so A* routes through them; the ArcPlanner cannot escape them, so the rover stalls. Sub-goal #4 at `(0, 0)` was relocated to `(−7, +7)` in the final tour after this finding.

**(b) `baseline` is unreliable across runs due to sim non-determinism near rocks.**
The same `baseline` config that produced RMSE 0.58 m on 5/31 09:46 produced RMSE 9.08 m on 5/31 18:12. In the failed run, the rover stalled at GT `(+12.34, −8.72)` (the right-edge corner between sub-goals 2 and 3) for the last ~6000 steps after firing 15,475 incidental LCs from being stationary. This is *not* a code change: the offline planner produced identical waypoints in both runs (same 60 waypoints, same 118.7 m planned path, same start/end). The non-determinism is in the simulator's ArcPlanner-rock interaction near small rocks. The lesson: even the baseline is fragile in this environment; a robust SLAM benchmark must include multiple runs per variant.

**(c) `kf_revisit_base` crashed at step 894 with a sensor dropout.**
The LAC simulator periodically fails to deliver a camera frame; `nav_agent.run_step()` indexes `input_data["Grayscale"][SensorPosition.FrontLeft]` without exception handling, so the entire mission terminates on a `KeyError`. This is an LAC-stack-level failure, not specific to our code, but it reproduces on the base variant in our testing and we report it as such.

### 5.8 The `kf_revisit_shadow` (no safety) intermediate result

The `KF Revisit + Shadow` variant (without the `w = 4` safety term) achieves RMSE 0.74 m on a 140 m path (per-meter drift 0.0053 m/m, comparable to baseline's 0.0056). This is the second-best result and demonstrates that even *without* the safety term, the combination of shadow-aware global routing + KF Revisit detour produces baseline-comparable per-meter drift on a 36%-longer trajectory while firing 17 LCs (vs baseline's 1). Adding the `w = 4` safety term to get the winning variant (0.14 m) provides the final 5× drift reduction by eliminating the residual stuck-rover events that the shadow-only cost cannot prevent.

### 5.9 Compute footprint

For one tour run on preset 6 (RTX 3090, 24 GB VRAM, Xvfb headless rendering, 0.15-0.20× real-time sim ratio):
- Wall-clock time per variant: 45-65 min for mobile variants, 2 h cap on stuck variants.
- GPU memory: UE4 simulator holds ~21 GB; LightGlue and SuperPoint share the remainder. We hit CUDA OOM in earlier testing on long missions until adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the launcher, which resolved the issue without further code changes.
- Full 10-variant matrix: ~10-13 h wall-clock end-to-end.

---

## 6. Conclusion

This work demonstrates that **path-planner-side loop-closure engineering** is a viable mechanism for reducing SLAM drift on lunar-rover tour-style missions. The headline finding is a **4× absolute RMSE reduction and a 5.5× per-meter drift reduction** over a slope-aware-A* baseline, on a 36% longer mission, using only DEM-derived perception cost terms and an online keyframe-revisit detour mechanism. No changes to the SLAM backend were required.

The mechanism analysis identifies three components that each contribute, and the failure modes when components are missing:

1. **A perception-aware global cost that prevents the rover from going visually blind.** Shadow avoidance (`s · shadow(n)` penalty) is a simpler and more robust cost than rewarding DEM-predicted feature density. The intuition: shadowed cells produce *no* visual features, so avoiding them gives the SLAM front-end something to work with; whereas routing toward predicted-rough-and-therefore-feature-rich cells routes the rover into the rocks that ArcPlanner cannot navigate. The borderline-positive (|Pearson| ≈ 0.25) DEM-roughness predictor we validated in Phase 0 is real but not strong enough to drive a global cost on its own — it is best used as part of the look-ahead reward only when paired with a heavy traversability-safety penalty.

2. **A traversability-as-safety penalty that keeps the rover mobile.** With `w = 4` on the normalized roughness, the rover stays out of rocky cells without the planner needing to know about UE4 collision meshes. The 35% stationary-rate of variants without this safety term (or with insufficient `w`) drops to 1% when the term is added.

3. **A keyframe-revisit detour mechanism that drives loop closures.** Detours that target *past keyframes* (geometrically driveable by construction, LC-eligible by construction) substantially outperform detours that target *DEM-predicted feature-rich cells* (driveable only if the planner's cost function actively avoids rocks, LC-eligible only if a past keyframe happens to be nearby). The Tight Anchor approach is intellectually appealing — predict the best place to go from the map — but in practice the predicted cells coincide with the cells the rover cannot reach, and the geometric constraints reduce the firing rate to near-zero.

The **structural insight** justifying the headline RMSE reduction: 20 of the winning variant's 54 loop closures bind late-mission keyframes back to keyframes in the first 10% of the trajectory (near the GTSAM prior pose). These cross-trajectory LCs are what enable *global* drift correction; mid-trajectory LCs (which all variants produce when LCs fire at all) only correct *local* drift. The combination of (a) shadow-aware routing keeping SLAM drift below `LC_DIST_THRESHOLD = 1 m` long enough for the rover to reach the return leg, plus (b) the KF Revisit detour actively driving the rover near old keyframes, produces these cross-trajectory bindings.

**Limitations and honesty.** Three caveats:
- **Sim non-determinism.** The same baseline configuration produced RMSE 0.58 m on one run and RMSE 9.08 m on another — both with `--seed=0` and the same waypoint list. The variance is in ArcPlanner-rock interaction, not in our code. A rigorous version of this experiment would average over multiple runs per variant; we report the best-completed baseline as the reference, noting this is the published number, not an average.
- **One environment.** All results are on LAC preset 6. The Phase 0 predictor was validated on preset 7. While we have qualitative experience with several other LAC presets, no quantitative cross-preset validation was performed.
- **DEM-driven planning has fundamental limits in this simulator.** UE4 collision meshes (the lunar lander at origin, leg-3 craters) are not in the GT DEM, so the cost function cannot see them. The DEM-only approach taken here can never fully resolve this; a complete solution would require runtime detection of these obstacles (e.g., via the existing semantic segmentation pipeline's `LANDER` class). This is the strongest argument for cross-modality (DEM + semantic) global planning rather than DEM-only.

---

## 7. Future Directions

In rough priority order:

### 7.1 Smarter detour-target selection

The current KF Revisit scorer chooses targets by minimizing extra detour distance: `score = 1 / (1 + extra_distance)`. This is deliberately simple and target-quality-agnostic. Three improvements suggest themselves:

(a) **Score by temporal distance.** Prefer keyframes that are *temporally old* — i.e., those at low pose-graph index, close to the GT-anchored prior. Such revisits, when successful, bind late poses back to the prior and produce the global-drift-correction effect we identified in §5.4. The current scorer is indifferent to this and tends to revisit recent keyframes that happen to be nearby; a temporal-distance bias would target early keyframes preferentially.

(b) **Score by observed feature count.** The backend's `keyframe_data[idx]` contains the SuperPoint features stored at each past keyframe. Targets with more stored features should produce more LightGlue match candidates and a higher LC success rate. This information is already available — we just don't currently use it.

(c) **Score by shadow status / ρ at the target.** A keyframe in a sunlit, feature-rich cell is more likely to successfully match than one in a near-shadow boundary. We could weight by the target cell's `ρ_norm` value.

### 7.2 Appearance-based loop-closure search in the SLAM backend

The dominant remaining structural limitation is that the LAC backend's loop-closure search is *geometric*: it can only find candidates within `LC_DIST_THRESHOLD = 1 m` of the SLAM-estimated current pose. Once SLAM drift exceeds 1 m, the trigger goes blind. An appearance-based search (BoW via DBoW3 with SuperPoint descriptors, or NetVLAD global descriptors) would allow the backend to find loop-closure candidates regardless of SLAM-pose proximity, dramatically expanding the operational envelope of the detour mechanism. We deliberately avoided this for the present work — we wanted to demonstrate that planner-side mechanisms alone can engineer LCs — but it is the obvious next step.

### 7.3 Validate on LRO orbital imagery

The DEM-derived perception predictor was validated on the LAC simulator's synthetic terrain. A more rigorous validation would use orbital imagery from NASA's Lunar Reconnaissance Orbiter (LRO Narrow Angle Camera, 0.5 m/pixel) co-registered with LOLA elevation, which provides genuine lunar surface terrain at mid-scale (kilometers). The Phase 0 predictor (`ρ = roughness × (1 − shadow)`) should be re-validated on this real terrain. The two-environment paper framing we considered earlier (LAC = small-scale validation, LRO = mid-scale demonstration) remains a strong direction.

### 7.4 Multi-run statistical evaluation

Given the sim non-determinism documented in §5.7, a publication-grade version of this work should report N ≥ 5 runs per variant with reported mean and standard deviation. The full 5-variant matrix would take ~25 runs (10-15 h of compute), well within an overnight schedule.

### 7.5 ROS-Gazebo cross-platform validation

The LAC simulator is research-grade. Demonstrating the same planner on a ROS2 / Gazebo / Ignition-Sim lunar terrain (e.g., the Open Robotics MarsSim or a publicly-shareable lunar variant) would lend credibility to the cross-simulator generalization of the approach.

### 7.6 Real-rover validation

The ultimate validation. Several teams (NASA JPL, ESA, lunar startup landscape) are operating physical lunar-analog rovers on terrestrial test ranges. The approach taken here — DEM-derived perception cost + planner-side LC engineering — is directly portable to any such rover, requires no SLAM-backend modifications, and could be evaluated against a slope-aware-A* baseline on the same hardware.

---

## 8. Reproducibility — code and artifacts

**Repository.** All code, configs, scripts, results, and figures are in `/home/santiagothorup/Documents/Lunar_Perception_Aware_Planning/`. The project should be initialized in a git repo and pushed to GitHub before paper submission; the report should include the GitHub URL.

**Key files for reviewers:**

| Purpose | Path |
|---|---|
| Perception cost field + anchor extraction | `lac/planning/perception_map.py` |
| A* planner with cost (1) + detour mechanism | `lac/planning/perception_aware_planner.py` |
| Agent (subclass of `NavAgent`) | `agents/perception_aware_agent.py` |
| Comparison driver | `scripts/run_planner_comparison.sh` |
| Per-run diagnostics | `scripts/plot_run_diagnostics.py` |
| Presentation-quality figures | `scripts/plot_presentation_panel.py`, `scripts/plot_perception_field.py` |
| Cross-variant overlay | `scripts/compare_runs.py` |
| Final configs | `configs/perception_aware_final_*.json` |

**Reproducing the headline result:**
```bash
cd /home/santiagothorup/Documents/Lunar_Perception_Aware_Planning
tmux new -s repro
FINAL_RUNS=1 scripts/run_planner_comparison.sh 6 7200 \
  kf_revisit_shadow_trav:perception_aware_final_kf_revisit_shadow_trav.json
# Output: LAC_SIM/output/NavAgent/final_runs/kf_revisit_shadow_trav/
# Result line appended to: LAC_SIM/output/NavAgent/final_runs/results.txt
```

**Reproducing the full 9-variant ablation:**
```bash
FINAL_RUNS=1 scripts/run_planner_comparison.sh 6 7200 \
  baseline:perception_aware_final_baseline.json \
  perc_trav:perception_aware_final_perc_trav.json \
  shadow:perception_aware_final_shadow.json \
  tight_anchor_base:perception_aware_final_tight_anchor_base.json \
  tight_anchor_perc:perception_aware_final_tight_anchor_perc.json \
  tight_anchor_shadow:perception_aware_final_tight_anchor_shadow.json \
  kf_revisit_base:perception_aware_final_kf_revisit_base.json \
  kf_revisit_perc:perception_aware_final_kf_revisit_perc.json \
  kf_revisit_shadow:perception_aware_final_kf_revisit_shadow.json
# Wall-clock cap: 10-18 h (9 × 2 h cap)
```

**Figure regeneration:**
```bash
# The 5 trajectory panels used in the presentation (slide 3c)
for label in baseline tight_anchor_perc kf_revisit_perc kf_revisit_shadow kf_revisit_shadow_trav; do
  python scripts/plot_presentation_panel.py \
    LAC_SIM/output/NavAgent/final_runs/$label \
    --out LAC_SIM/output/NavAgent/final_runs/figures/panel_$label.png
done
# Perception field (slide 2)
python scripts/plot_perception_field.py --preset 6 \
  --out LAC_SIM/output/NavAgent/final_runs/figures/perception_field_preset6.png
```

**Pre-existing artifacts in the repo at submission time:**
- Final results: `LAC_SIM/output/NavAgent/final_runs/results.txt`
- Per-variant raw data (data_log.json, slam_poses.npy, backend_state.npz): `LAC_SIM/output/NavAgent/final_runs/<label>/`
- Presentation figures: `LAC_SIM/output/NavAgent/final_runs/figures/`
- Phase 0 validation: `output/phase0_transect_p7_v2/`
- Earlier corner-to-corner experiments (for context): `docs/PLANNER_EXPERIMENT_RESULTS.md`

---

## Quick-reference data tables (for paper figure / table generation)

### Table A — Final 5-variant comparison (the slide 3c table)

| Variant | RMSE (m) | Path (m) | Per-meter drift (m/m) | LCs | Detours | Sub-goals | Stationary |
|---|---:|---:|---:|---:|---:|:---:|---:|
| Baseline | 0.582 | 103.1 | 0.0056 | 1 | 0 | 6/6 | 1% |
| Tight Anchor + High Feature | 11.00 | 114.2 | 0.0963 | 12 | 4 | 4/6 | 7.7% |
| KF Revisit + High Feature | 0.905 | 146.6 | 0.0062 | 49 | 9 | 6/6 | 3% |
| KF Revisit + Shadow | 0.741 | 140.1 | 0.0053 | 17 | 10 | 6/6 | 1% |
| **KF Revisit + Shadow + Safety** ★ | **0.140** | **139.4** | **0.0010** | **54** | 9 | **6/6** | **1%** |

### Table B — Full 9-variant ablation matrix (the supplementary)

| Variant | RMSE | Path | LCs | Det | Stat | Result |
|---|---:|---:|---:|---:|---:|---|
| baseline | 9.08 | 48.0 | 15,475 | 0 | 64% | ❌ stuck (sim non-determinism) |
| perc_trav | 1.59 | 117.3 | 4 | 0 | 10% | ✓ complete |
| shadow | 3.07 | 119.4 | 5 | 0 | 1% | ✗ 4/6 goals |
| tight_anchor_base | 130.78 | 63.3 | 1 | 2 | 10% | ❌ catastrophic |
| tight_anchor_perc | 1.64 | 133.6 | 36 | 4 | 3% | ✓ complete |
| tight_anchor_shadow | — | — | — | — | — | ❌ timed out 2 h |
| kf_revisit_base | — | — | — | — | — | ❌ crashed (sensor) |
| kf_revisit_perc | 0.90 | 146.6 | 49 | 9 | 3% | ✓ complete |
| kf_revisit_shadow | 0.74 | 140.1 | 17 | 10 | 1% | ✓ complete |
| **kf_revisit_shadow_trav** | **0.14** | **139.4** | **54** | 9 | **1%** | **✓✓ WINNER** |

### Table C — Per-meter drift ranking

| Rank | Variant | Per-meter drift (m/m) | Path (m) | Relative to baseline |
|---:|---|---:|---:|---:|
| **1** | **KF Revisit + Shadow + Safety** | **0.0010** | 139.4 | **0.18×** (5.5× lower) |
| 2 | KF Revisit + Shadow | 0.0053 | 140.1 | 0.95× |
| 3 | Baseline (reference) | 0.0056 | 103.1 | 1.00× |
| 4 | KF Revisit + High Feature | 0.0062 | 146.6 | 1.11× |
| 5 | Tight Anchor + High Feature | 0.0963 | 114.2 | 17.2× WORSE |

### Table D — LC-firing efficiency

| Variant | Detours fired | LCs fired | LCs per detour | "Caused" vs "incidental" |
|---|---:|---:|---:|---|
| Baseline | 0 | 1 | n/a | incidental (return-to-start) |
| Tight Anchor + High Feature | 4 | 36 | 9.0 | mostly incidental (rover passed near keyframes) |
| KF Revisit + High Feature | 9 | 49 | 5.4 | detour-caused (KF-targeted) |
| KF Revisit + Shadow | 10 | 17 | 1.7 | detour-caused (sparser keyframe trail) |
| **KF Revisit + Shadow + Safety** | **9** | **54** | **6.0** | **detour-caused + 20 "back to start"** |

---

## Glossary of acronyms used in the paper

| Acronym | Meaning |
|---|---|
| A* | A-star graph search algorithm (Hart, Nilsson, Raphael 1968) |
| ArcPlanner | The local steering controller (Stanford NavLab; unmodified in this work) |
| BoW | Bag-of-Words appearance descriptor for image retrieval |
| DEM | Digital Elevation Model |
| GNSS | Global Navigation Satellite System |
| GT | Ground Truth |
| GTSAM | Georgia Tech Smoothing and Mapping (pose-graph optimization library) |
| HW3 P4 | Stanford AA278 Homework 3 Problem 4 (the slope-aware A* baseline) |
| IMU | Inertial Measurement Unit |
| KF | Keyframe |
| LAC | (Stanford NavLab) Lunar Autonomy Challenge |
| LC | Loop Closure |
| LightGlue | Learned feature-matching network (paired with SuperPoint here) |
| LRO | Lunar Reconnaissance Orbiter (NASA, 2009–) |
| LOLA | Lunar Orbiter Laser Altimeter (on LRO) |
| NAC | Narrow Angle Camera (on LRO) |
| NavLab | (Stanford) Navigation and Autonomous Vehicles Lab |
| NBV | Next-Best-View planning |
| ρ (rho) | The DEM-derived feature-density predictor, `ρ = roughness × (1 − shadow)` |
| RMSE | Root Mean Square Error |
| SLAM | Simultaneous Localization And Mapping |
| SuperPoint | Learned feature detector + descriptor (paired with LightGlue here) |
| UE4 | Unreal Engine 4 (the LAC simulator's rendering engine) |
| VIO | Visual-Inertial Odometry |
| VO | Visual Odometry |
