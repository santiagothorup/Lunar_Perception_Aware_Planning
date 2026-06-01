# Report Background — Perception-Aware Path Planning for Lunar Rover Autonomy

> Source material for the final report (sections 1-4: Introduction, Related Work, Problem Statement, Approach). All numerical values verified against the final-run results in `LAC_SIM/output/NavAgent/final_runs/`. Section 5 (Results) lives in `REPORT_RESULTS.md`.

---

## 1. Introduction

### 1.1 Motivation

Autonomous lunar rovers operating on the lunar south pole cannot rely on global navigation satellite systems (GNSS) — GPS does not reach the Moon and Lunar GNSS infrastructure does not yet exist. Localization is therefore derived from on-board sensing, typically a fusion of visual odometry (VO) over stereo imagery, inertial measurement units (IMU), and wheel encoders. The Stanford NavLab Lunar Autonomy Challenge (LAC) framework adopted in this work uses a stereo-camera SLAM pipeline (SuperPoint + LightGlue feature matching, GTSAM pose-graph backend) as the canonical example of this localization stack.

Visual-inertial SLAM is known to accumulate ~0.5–1% positional drift per meter traveled in well-instrumented terrestrial conditions; on the lunar surface, where the terrain is broadly uniform (regolith with sparse rocks), shadows are deep and high-contrast, and the sun grazes the horizon at the south pole (altitude < 2°), drift is empirically worse. Multi-kilometer surface traverses planned for upcoming missions (e.g., NASA VIPER, IM-2, future Artemis-era ops) would accumulate tens of meters of localization error if unmitigated — well beyond the sub-meter tolerance required for sample-return and infrastructure-emplacement tasks.

The standard mitigation is **loop closure**: when the rover re-observes a previously mapped region, the SLAM backend can match current sensor data against an earlier keyframe and add a relative-pose constraint to the pose graph. This constraint allows the optimizer (Levenberg–Marquardt over GTSAM `BetweenFactorPose3` factors in our backend) to globally redistribute accumulated drift. Loop closures are the only mechanism in our pipeline that *bounds* SLAM error globally rather than letting it grow monotonically with distance.

However, loop closures are not free: they require the rover to *physically revisit* terrain that is (a) visually distinctive enough to produce successful feature matches, and (b) geometrically close enough to a stored keyframe for the backend's proximity-based candidate-pair search to find it. **A rover whose mission plan never revisits its own trail will fire few or no loop closures, and its localization will drift unbounded.**

This work asks: *can the global path planner be made loop-closure aware?* If we can route the rover through regions that are (i) predicted to be visually distinctive and (ii) close enough to past trajectory for re-encounter, we should be able to lift the loop-closure rate without sacrificing mission objectives, and reduce SLAM drift correspondingly.

### 1.2 Significance

Three reasons this matters for near-term lunar operations:

1. **Multi-kilometer traverses are imminent.** VIPER's planned routes exceed 20 km of total surface drive; even sub-1% drift accumulates to hundreds of meters of localization error without mitigation. Active loop-closure engineering, baked into the planner, is one of the few software-only mitigations available.
2. **Heritage planners ignore perception.** The dominant lunar rover global planner archetype (slope-aware A* over a digital elevation model, e.g., as in Stanford's AA278 Homework 3 Problem 4 baseline used here) optimizes for traversability and energy, not perceptibility. The planner cannot tell the rover *"this leg is going to leave you blind"* until the rover gets there and SLAM has already drifted.
3. **The lunar terrain is uniquely hostile to perception.** Lunar regolith is texture-poor in shadow (no atmosphere → near-zero illumination); the sun is a point source at low altitude, producing long, deep shadows that move predictably with time of day; rocks are sparse but visually rich and create the only landmarks. These constraints make the problem unusually well-suited to a DEM-driven prediction: roughness and shadow are the two features the planner *can* compute a priori from a global map.

### 1.3 Approach summary

This work proposes:
1. A **perception-aware A\* global planner** whose edge cost extends slope-aware A* with terms for (a) DEM-derived feature-density prediction `ρ = roughness × (1 − shadow)`, (b) traversability penalty `w · rough(n)` for ArcPlanner-safety, and (c) shadow-avoidance `s · shadow(n)` as a simpler perception cost.
2. An **online loop-closure detour mechanism** that, when SLAM has accumulated `steps_since_last_LC > threshold` updates without a successful loop closure, splices a short detour to a *past keyframe* (the *KF Revisit* tactic) into the global plan, then resumes the planned tour.
3. An **ablation matrix** crossing three global cost configurations (slope-only baseline, feature-density-rewarding `High Feature`, shadow-avoiding `Shadow`) with three detour modes (none, `Tight Anchor` to pre-computed feature-rich cells, `KF Revisit` to past keyframes).

The headline finding is that the combination *Shadow + Safety global cost + KF Revisit detour* achieves SLAM RMSE of **0.14 m on a 139.4 m perimeter tour**, versus **0.58 m on a 103 m tour** for the slope-aware baseline — a **4× reduction in absolute RMSE on a 36% longer trajectory** and a **5.5× reduction in per-meter drift** (0.0010 vs 0.0056 m / m). The mechanism: 54 loop closures fired during the run, of which 20 bound late-mission keyframes back to keyframes in the first 10% of the trajectory (near the GTSAM prior pose), providing genuine global drift correction.

---

## 2. Related Work

### 2.1 Visual-Inertial SLAM and Loop Closure

Modern visual-inertial SLAM systems — ORB-SLAM3, VINS-Fusion, OpenVINS, Kimera — share a common architecture: a tightly-coupled VIO front-end (feature extraction + tracking + bundle adjustment over a local window) feeds a global pose-graph back-end (factor graph optimized by iSAM2 or GTSAM Levenberg-Marquardt) that incorporates loop closure constraints. The LAC reference SLAM used in this work follows this archetype: stereo SuperPoint features + LightGlue matcher in the front-end, GTSAM pose-graph back-end with `BetweenFactorPose3` factors for both VO odometry and loop closure. Loop-closure detection in the LAC backend is a *geometric* proximity check: if the SLAM-estimated current pose is within `LC_DIST_THRESHOLD = 1 m` and `LC_ANGLE_THRESHOLD = 5°` of a stored keyframe (excluding the most recent 10), the backend attempts a LightGlue feature match; if `≥ 100` matches survive at score `≥ 0.5`, a relative-pose factor is added with diagonal noise sigmas `[0.00087, 0.00087, 0.00087, 0.005, 0.005, 0.005]` (rotation in radians, translation in meters).

Appearance-based loop-closure search (Bag-of-Words via DBoW2/DBoW3, NetVLAD descriptor matching) is the prevailing alternative, used in ORB-SLAM3 and others. It is more robust to SLAM drift (an appearance-based candidate search can fire even when SLAM thinks the rover is far from any old keyframe) but adds dependencies (descriptor pipeline, vocabulary training) we did not implement to keep the SLAM stack unchanged. The implication for this work is important: **the LAC backend's geometric LC trigger is robust only as long as SLAM stays within `LC_DIST_THRESHOLD` of ground truth**. Once accumulated drift exceeds 1 m, the trigger cannot detect a true revisit — a structural limit our planner-side mechanism is designed to work *within* rather than fix.

### 2.2 Active and Perception-Aware SLAM

Carrillo *et al.* (2012) formalize **active SLAM** as a problem of choosing control actions to maximize expected information gain about the map and pose. The standard objective is D-optimality (maximize the determinant of the posterior pose covariance's inverse), which monotonically tightens with each new measurement weighted by its information content. Subsequent work generalizes to coverage-aware exploration (Stachniss *et al.*), next-best-view planning for 3D reconstruction (e.g., Bircher *et al.* receding-horizon NBV), and belief-space planning (Platt *et al.*, Indelman *et al.*). The unifying premise: route the robot through regions whose measurements will most reduce uncertainty.

The closest prior to this work is the family of **perception-aware** planners that augment a traversability cost with a feature-richness reward computed from imagery or a learned predictor (Mostegel *et al.*, Costante *et al.*, Zhang & Scaramuzza, and recent work in lunar VO/SLAM by JPL and ESA groups). These typically operate either:
- *Locally* (the planner considers visible features ahead of the rover, e.g., next-best-view), or
- *Over a pre-mapped environment* (a feature-density grid built from a previous traverse).

This work differs in that the feature-density prediction is computed *a priori* from a Digital Elevation Model (DEM) — specifically, the ground-truth heightmap that LAC's mission control provides — without requiring any prior visual observation of the terrain. This is appropriate for lunar planning because the DEM is the *only* a-priori knowledge available; orbital imagery (LRO NAC) provides comparable terrain knowledge for actual lunar missions.

### 2.3 Lunar Rover Localization and Planning

Lunar rover navigation literature (Carsten *et al.*, Krüsi *et al.*, recent JPL VIPER traverse-planning publications) acknowledges drift as the dominant localization risk, with mitigations historically focused on (i) periodic stop-and-localize against orbital imagery, (ii) IMU-aided VO, or (iii) operator-in-the-loop waypoint adjustment. None to our knowledge treat **the global planner itself as a loop-closure engineering tool**. The Stanford NavLab AA278 Homework 3 Problem 4 slope-aware A* planner (`alpha=2`, `theta_ref=10°`, slope-gated at `theta_max=20°`) used as our baseline is representative of operational lunar planners.

### 2.4 Where this work sits

This work occupies a specific niche: a **path-planner-side loop-closure mechanism** that
- requires no changes to the SLAM backend,
- requires no new perception pipeline beyond what the SLAM front-end already runs,
- requires only DEM information (universally available for any planned lunar surface mission),
- is composable with any traversability-aware A*.

The two principal contributions are: (1) demonstrating that **shadow avoidance** is a simpler, more robust DEM-derived perception cost than the natural alternative of routing toward predicted-feature-rich (high-`ρ`) cells, and (2) showing that an **online keyframe-revisit detour** is a sufficient mechanism to actively engineer loop closures in a tour-style mission, achieving 4× lower SLAM RMSE than a baseline that lacks this mechanism.

---

## 3. Problem Statement

### 3.1 Environment and assumptions

- A rover navigates on a 2D ground plane parameterized by world coordinates `(x, y) ∈ [-MAP_EXTENT, MAP_EXTENT]²` with `MAP_EXTENT = 13.5 m` (the LAC sim's 27 m × 27 m operational area). Discretization is `CELL_WIDTH = 0.15 m` (180 × 180 cells).
- The rover has access to a ground-truth **Digital Elevation Model** `Z : ℝ² → ℝ` defined on the same grid, plus a known sun azimuth and altitude (LAC: `az = 263.575°` CCW from +X, `alt = 1.488°` — both fixed across all presets).
- The rover's pose is estimated by an on-board SLAM system whose state is a sequence of keyframe poses `{X_i}` in a pose graph `G = (V, E)`. Edges include VO odometry factors (consecutive keyframes) and loop-closure factors (geometrically detected revisits). The SLAM estimate `X̂_i` may differ from the true pose `X_i*` by an accumulated drift `‖X̂_i − X_i*‖_2`.
- The mission is a *tour*: a sequence of `M` ordered sub-goals `G = (g_1, …, g_M)` with `g_k ∈ ℝ²`. The rover must visit each sub-goal in order, starting from a spawn pose `x_s` (in our experiments, `g_1 = g_M = (−12, −12)` so the tour is a closed loop).

### 3.2 Planning problem

The global planner chooses a sequence of waypoints `W = (w_0, w_1, …, w_N)` with `w_0 = x_s`, `w_N = g_M`, and a contiguous A* graph search between each consecutive sub-goal. The local controller (the unchanged LAC ArcPlanner) is responsible for safe steering between waypoints and is treated as a black box.

The planner's edge cost between adjacent grid cells `n_1 = (r_1, c_1)`, `n_2 = (r_2, c_2)` (where indexing uses cell row/column with `n.xy ∈ ℝ²` the cell-center world coordinate) is:

```
c(n_1, n_2)  =  d · ( 1
                    + α · (θ(n_2)/θ_ref)²
                    + β · (1 − ρ(n_2))
                    + w · trav(n_2)
                    + s · shadow(n_2) )                                  (1)
```

where
- `d = ‖n_2.xy − n_1.xy‖_2` is the Euclidean distance,
- `θ(n_2)` is the local slope of the DEM at `n_2`, computed from gradients `Z_x, Z_y` via `arctan(√(Z_x² + Z_y²))`,
- `θ_ref = 10°` is a reference slope, and an additional **hard infeasibility** filter rejects neighbors with `θ > θ_max = 20°`,
- `α = 2` is the slope penalty weight (P4 baseline default),
- `ρ(n) = ρ_lookahead(n)` is the **mean predicted feature density at three look-ahead distances along the entry direction** (1.5, 2.5, 3.5 m from `n_2` along `n_1 → n_2`), where the underlying field is `ρ_norm = roughness × (1 − shadow)` normalized by the 95th-percentile per map,
- `trav(n)` is the normalized roughness at the entered cell, `roughness / p95(roughness)`, clamped to [0, 1],
- `shadow(n)` is the binary shadow indicator: 1 if the cell receives no direct sunlight (computed by a per-cell ray-cast toward the sun direction at altitude `alt`), 0 otherwise,
- `(β, w, s)` are configurable weights that select between the three global-cost variants studied (see §4.1).

The multiplier `(1 + ...)` is always `≥ 1` and the Euclidean distance heuristic remains admissible for A*.

A* is run between consecutive sub-goals `g_k → g_{k+1}` and the resulting paths are concatenated and resampled to a uniform waypoint spacing of 2 m, producing the final waypoint list `W`. The planner exposes `get_waypoint(step, pose)` to the agent as a drop-in replacement for the existing `WaypointPlanner` interface.

### 3.3 SLAM, loop closure, and the localization metric

The SLAM backend maintains a pose graph that grows incrementally. On each new keyframe, the backend (a) adds a VO odometry factor connecting it to the previous keyframe, (b) checks for **loop-closure candidates** by searching for any earlier keyframe (excluding the most recent `LOOP_CLOSURE_EXCLUDE = 10`) whose SLAM-estimated XY position is within `LC_DIST_THRESHOLD = 1.0 m` and whose orientation differs by less than `LC_ANGLE_THRESHOLD = 5°`. For each candidate, the backend runs LightGlue against the candidate's stored features; if `≥ LC_MIN_MATCHES = 100` matches at score `≥ LC_MIN_SCORE = 0.5` are found, a `BetweenFactorPose3` is added with diagonal noise sigmas `[0.00087rad × 3, 0.005m × 3]`. The graph is then re-optimized by Levenberg-Marquardt.

Note that the LC search uses the **SLAM-estimated** pose, not ground truth: if the SLAM estimate has drifted by more than 1 m, the trigger cannot detect a true revisit. This bounds the effectiveness of the LC mechanism under accumulated drift.

The end-of-mission localization error is computed as the RMSE between the final SLAM trajectory and the synchronized ground-truth poses:
```
RMSE  =  √( (1/N) Σᵢ ‖X̂ᵢ − Xᵢ*‖_2² )                                     (2)
```
(LAC implementation: `positions_rmse_from_poses(slam_poses, slam_eval_poses)` in `lac/util.py`.) Per-meter drift `RMSE / path_length` is reported as a secondary metric that normalizes for trajectory length, since absolute RMSE on a 50 m path is not comparable to RMSE on a 150 m path.

### 3.4 Loop-closure detour mechanism

In addition to the modified global cost, we introduce an **online detour mechanism** that triggers when the SLAM backend has not produced a loop closure in the last `T_LC = 400` keyframe-updates (controlled by the `steps_since_lc_threshold` parameter). On trigger, the planner selects a **detour target** `τ ∈ ℝ²` using one of two tactics:

**Tactic A — Tight Anchor.** Pre-compute a set of `K ≤ 12` *anchor cells* `A = {a_1, …, a_K}` from the DEM-derived `ρ_norm` field: anchors are local maxima of `ρ_norm` (5-cell footprint) above a threshold `ρ_min = 0.4`, filtered to (a) be slope-feasible (`θ(a) ≤ θ_max`), (b) have local roughness ≤ `roughness_max = 0.5`, (c) have local rock density ≤ `rock_density_max = 0.01` in a 1 m window, and (d) be separated from other anchors by ≥ `min_separation_m = 4 m`. At runtime, the detour target is the anchor that (i) lies in the 2-8 m proximity band of the current pose, (ii) is near a past keyframe (within `1.5 × LC_DIST_THRESHOLD`), and (iii) minimizes the additional path cost to the next sub-goal.

**Tactic B — KF Revisit.** Score over the set of past keyframes (excluding the most recent `LOOP_CLOSURE_EXCLUDE = 10`) that lie within `[2 m, 8 m]` of the current pose, choosing the keyframe that minimizes the additional path length `‖pose → kf‖ + ‖kf → next_subgoal‖ − ‖pose → next_subgoal‖`.

Once a target is chosen, the planner **splices** the global plan: it computes a new A* path `pose → τ → g_{current} → g_{current+1} → … → g_M` through all remaining sub-goals, resamples to waypoint spacing, and replaces `W` from the current index onward. The detour is considered "verified" if a loop closure fires with its anchor keyframe within `2 × LC_DIST_THRESHOLD` of `τ` before either (a) a per-leg attempt cap of 3 is reached, or (b) `MAX_DETOUR_STEPS = 1500` steps pass without verification (in which case the target is added to a per-leg blacklist and the rover resumes its tour).

The choice between Tactic A and Tactic B is a design parameter. Tactic A targets locations *predicted* by the DEM to be feature-rich; Tactic B targets locations the rover has *empirically* visited (so revisit driveability is guaranteed by construction). Section 4 motivates the choice; Section 5 (Results) shows that Tactic B substantially outperforms Tactic A in our environment.

---

## 4. Approach

### 4.1 Three global-cost configurations

Equation (1) parameterizes a family of A* costs; we evaluate three concrete settings, defined by the `(α, β, w, s)` quadruple. All share `α = 2` and `θ_ref = 10°` (P4 defaults), differing in `(β, w, s)`:

| Variant | β (ρ reward) | w (rough penalty) | s (shadow penalty) | Intended behavior |
|---|---|---|---|---|
| **Baseline** (HW3 P4 slope-aware A*) | 0 | 0 | 0 | shortest slope-feasible path |
| **High Feature** | 2 | 8 | 0 | route toward predicted feature-rich cells, with strong rough-cell safety penalty |
| **Shadow Avoidance** | 0 | 0 | 2 | route through sunlit cells only; no roughness component |

The High Feature variant's `w = 8` rough-cell penalty is necessary to prevent the rover from being routed into rocky cells: as Phase 0 validation showed, high-ρ cells are positively correlated with roughness (the very source of the "feature-density predictor" — see §4.3). Without the penalty, the rover stalls; with the penalty, the rover routes along the *edges* of feature-rich corridors while still capturing elevated look-ahead-ρ exposure.

The Shadow Avoidance variant drops the ρ term entirely and relies only on the shadow indicator. This is conceptually the simplest cost — *prefer sunlit cells* — and avoids the High Feature variant's reliance on the roughness predictor (which we found to be a borderline-positive signal, not strong enough to be the primary cost driver).

### 4.2 The detour tactic comparison

For each global cost we additionally test three detour modes: `none` (no online detour), `Tight Anchor` (Tactic A), and `KF Revisit` (Tactic B). The full evaluation is a 3 × 3 = 9-cell ablation matrix (§5).

The hypothesis distinguishing Tight Anchor from KF Revisit:

- **Tight Anchor** *predicts* good revisit locations from the DEM. If the predictor is accurate, this should consistently route to high-feature cells. *Risk*: anchor cells often coincide with rocky cells (see §4.3), so the rover may stall trying to reach them; the pre-computed set is also limited in number (≤ 12), so the geometric proximity filter often eliminates all candidates and the detour no-ops.
- **KF Revisit** *guarantees* a driveable target (the rover already went there) and *guarantees* a loop-closure-eligible target (the backend already stored a keyframe there). *Risk*: the rover only revisits places it has already been, so the detour cannot route to fundamentally new feature-rich regions.

### 4.3 Phase 0 — DEM-derived feature-density predictor (a priori validation)

Before evaluating the planner end-to-end, we validated whether the DEM-derived feature-density predictor `ρ = roughness × (1 − shadow)` is in fact a useful signal for feature count.

**Procedure.** A safe collector agent (slope-aware A* + ArcPlanner + rock-frontend + stuck-detection-and-backup) traversed a serpentine raster across LAC preset 7, capturing `n = 10,565` clean stereo frames. For each frame, we computed (a) the number of SuperPoint features detected by the SLAM frontend, and (b) four candidate predictor values: pose-ρ (cell the rover is on), look-ahead-ρ at distances `d ∈ {1.5, 2.5, 3.5, 4.0} m` ahead of the rover.

**Result.** Across hypotheses H1–H4, the strongest correlate was **look-ahead ρ at `d = 1.5 m`**, with `|Pearson| = 0.20–0.29` (preset 7 cleanest at 0.29). Mean feature count rose monotonically with roughness quartile (Q1: 658 features, Q4: 757 features, ~15% gain), confirming the signal. Pose-ρ *anti*-correlated with features — a finding that drove the implementation choice to use **look-ahead-ρ** (the predicted feature density of the cell the camera is *looking at*, not the cell the rover is *on*) as the cost term.

**Interpretation.** `|Pearson| ≈ 0.25` is a **borderline-positive** signal: the predictor captures real structure, but is not strong enough to use as a hard requirement. This motivated the High Feature variant's heavy `w = 8` traversability penalty (to keep the predictor from steering the rover into unsafe cells) and the eventual addition of the Shadow Avoidance variant (a simpler signal that does not rely on the borderline-strength roughness predictor).

### 4.4 Detour mechanism — design details

**Trigger.** A SLAM-state provider closure is injected into the planner from the agent, exposing `{pose_idx, loop_closures, keyframe_xy_lc_eligible, keyframe_xy_all}` on each `get_waypoint()` call. The detour mechanism fires when `pose_idx − last_lc_step > steps_since_lc_threshold = 400`.

**Target selection (KF Revisit).** Among LC-eligible past keyframes within the 2-8 m proximity band of the current SLAM pose, select the keyframe minimizing `Δd = ‖pose → kf‖ + ‖kf → next_subgoal‖ − ‖pose → next_subgoal‖` (the additional distance the detour costs). Skip keyframes in the per-leg `_detour_history` (a blacklist of recently-failed targets).

**Splice mechanics.** When a target `τ` is selected, the planner replaces `self.waypoints` from the current index onward with a freshly-planned A* path through `pose → τ → g_k → g_{k+1} → … → g_M`, where `g_k` is the next un-reached sub-goal. The first sample of the new path is dropped (it lies at current pose, would be an instant advance), `waypoint_idx` is reset to 0, and `last_waypoint_step` is reset to the current step. To track the global sub-goal counter across re-plans, a `_subgoal_index_offset` is set to the current sub-goal index so the sub-goal advance logic correctly maps the new local indices back to the global tour.

**LC verification and timeout.** After a detour fires, `_detour_active` is set, the current `len(loop_closures)` is recorded as the baseline, and `_detour_target_xy` is stored. On each subsequent step, if `len(loop_closures)` increases, the newly fired LC's anchor keyframe is checked: if its SLAM-estimated XY position is within `2 × LC_DIST_THRESHOLD = 2 m` of the detour target, the detour is marked verified-success and `_detour_active` is cleared. Otherwise (or if `step − _detour_start_step > MAX_DETOUR_STEPS = 1500`), the detour is marked failed, the target is added to `_detour_history`, and the rover continues.

**Per-leg attempt cap.** To prevent unbounded detour loops on a difficult leg, at most `MAX_DETOUR_ATTEMPTS_PER_LEG = 3` detours may fire while the same sub-goal is the active target. The counter resets when the sub-goal advances.

**Waypoint-timeout widening.** During an active detour, the per-waypoint stuck-detection timeout is widened from `WAYPOINT_TIMEOUT = 2000` to `WAYPOINT_TIMEOUT_DETOUR = 4000` steps, because detour legs typically involve more ArcPlanner replanning around rocks and benefit from a longer recovery budget.

### 4.5 The tour mission used for evaluation

To expose meaningful dead-reckoning drift while keeping mission length comparable across variants, the evaluation mission is a **6-sub-goal perimeter tour** on LAC preset 6 (Moon_Map_01, MISSIONS_SUBSET=6):

```
spawn ≈ (−3.5, −6.9)
  → g_1 = (−12, −12)      bottom-left corner
  → g_2 = (+10, −12)      bottom-right
  → g_3 = (+12, +12)      top-right
  → g_4 = ( −7,  +7)      NW interior pivot (intentional, to skirt UE4 collision meshes on the diagonal)
  → g_5 = (−10, +12)      top-left
  → g_6 = (−12, −12)      return to start (closes the loop)
```

The tour planned-path length is ~117 m (≈ 2× the corner-to-corner baseline of 53 m used in earlier Phase-1 testing), which gives the baseline a measurable drift budget. The intermediate sub-goal `g_4 = (−7, +7)` was added after testing showed that the leg-3-to-leg-5 diagonal `(+12, +12) → (−10, +12)` routed through a region of UE4-mesh craters that the DEM-based cost cannot see, causing stalls. With `g_4` in place, the path safely traverses the upper-half region.

The original goal of including a central crossing waypoint (`(0, 0)`) was abandoned after we discovered that the lunar lander UE4 collision mesh sits at world origin and deadlocks the rover regardless of cost-function tuning. We use this finding in a side-note on UE4-mesh limitations of DEM-driven planning (§5).

### 4.6 Implementation summary

The work consists of three new modules added on top of the unmodified LAC SLAM stack:

| Module | File | Purpose |
|---|---|---|
| `PerceptionMap` | `lac/planning/perception_map.py` | DEM → (roughness, shadow_mask, ρ_norm, anchors); accessors `get_lookahead_density`, `get_traversability_cost`, `get_shadow_value`; `compute_anchors(dem)` for Tactic A |
| `PerceptionAwarePlanner` | `lac/planning/perception_aware_planner.py` | A* with cost (1); multi-sub-goal stitching via `_plan_sequence`; KF Revisit / Tight Anchor target selection; `_splice_detour`; sub-goal-index-offset tracking |
| `PerceptionAwareAgent` | `agents/perception_aware_agent.py` | Subclasses `NavAgent`; instantiates `PerceptionMap`, computes anchors, constructs `PerceptionAwarePlanner`, injects SLAM provider; overrides `finalize()` to save sidecar JSON with anchors, detour events, and planner configuration |

The SLAM backend (`lac/slam/backend.py`), GTSAM factor graph wiring (`lac/slam/gtsam_util.py`), feature tracker (`lac/slam/feature_tracker.py`), and ArcPlanner (`lac/planning/arc_planner.py`) are all unchanged from the LAC reference implementation. This was a deliberate design constraint: we wanted to demonstrate that the planner-side mechanism alone, without SLAM modifications, can engineer loop closures.

Reproducibility tooling: `scripts/run_planner_comparison.sh` drives multi-variant comparison runs under a wall-clock cap; `scripts/plot_run_diagnostics.py` and `scripts/plot_presentation_panel.py` produce per-run trajectory overlays; `scripts/compare_runs.py` produces cross-variant comparison figures.

---

## Configurations used in the final evaluation

All final-run configs live in `configs/perception_aware_final_*.json`. The five used in the headline results table:

| Variant | β | w | s | detour mode |
|---|---|---|---|---|
| `final_baseline.json` | 0 | 0 | 0 | none |
| `final_tight_anchor_perc.json` | 2 | 8 | 0 | Tight Anchor |
| `final_kf_revisit_perc.json` | 2 | 8 | 0 | KF Revisit |
| `final_kf_revisit_shadow.json` | 0 | 0 | 2 | KF Revisit |
| **`final_kf_revisit_shadow_trav.json`** | 0 | 4 | 6 | KF Revisit (winner) |

Common detour parameters across all detour variants: `steps_since_lc_threshold = 400`, `max_detour_steps = 1500`, `max_detour_attempts_per_leg = 3`, `waypoint_timeout_detour_steps = 4000`, `detour_keyframe_min_m = 2`, `detour_keyframe_max_m = 8`, `LC_DIST_THRESHOLD = 1.0 m`.
