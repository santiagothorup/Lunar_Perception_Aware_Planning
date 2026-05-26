# Project Turnover — Lunar Perception-Aware Planning

> **Purpose**: This document is the handoff packet for any AI agent or human picking up this project mid-stream. Read this first. Then read `Project_Background.md` (technical design) and `SIM_STARTUP.md` (install procedure). The three files together fully specify the project's intent, status, work product, and immediate next steps.

---

## 1. Project at a glance

**Course**: AA278 Lunar PNT, Stanford, Spring 2026. Solo final project by Santiago Thorup.
**Deadlines**: slides 2026-06-01 (in-class), report 2026-06-05 (4 PM PT, ION format, 6–8 pages).
**Status as of 2026-05-26**: Phase 0 validation complete (results in `output/phase0_validation*/`). **Vulkan-on-WSL UNBLOCKED**: kisak-mesa PPA on Ubuntu 24.04 ships Mesa 26.1+ with the `dzn` driver, which exposes the NVIDIA RTX 4060 as `DRIVER_ID_MESA_DOZEN` via D3D12. `vkcube` confirmed rendering on the dGPU. UE4 should now run. **Next concrete step**: replay the env install in the 24.04 distro (per `SIM_STARTUP.md` Sections 1.1, 2, 3, 4) and launch the sim. No planner code written yet (deliberate — gated on richer Phase 0 results).

**One-sentence project description**: Build a **goal-to-goal** trajectory planner for a lunar rover that proactively routes through visually feature-rich terrain to minimize SLAM localization error, combining DEM-derived roughness + sun shadow casting into a feature-density predictor used as a soft cost in A*.

**Baseline comparison**: Slope-aware A* from AA278 HW3 Problem 4 (the implementation is at `data/Example_Implementations/HW3_Final/AA278_2026_HW3_P4.ipynb` and `data/Example_Implementations/HW3_Final/supplemental/dem.py` + `supplemental/util.py`). The perception-aware planner extends P4's cost function with an additive ρ-weighted term.

**Primary metric**: SLAM RMSE vs. ground truth over a planned path (`positions_rmse_from_poses` already in `lac/util.py`).

**Two evaluation environments**:
1. **Mid-scale (LRO real lunar tile)**: `data/Example_Implementations/HW3_Final/data/dem_tile.npz` (2000 × 2000 cells at 5 m/pixel = 10 km × 10 km, real LRO LOLA south-pole DEM). No SLAM here — algorithmic + visual demo of paths on actual lunar terrain.
2. **Small-scale (LAC simulator)**: 27 m × 27 m simulator environment (Moon_Map_01) with ~10 preset variations. Hand-picked (start, goal) pairs. RMSE measured end-to-end.

---

## 2. Where we left off (read this carefully)

### What's done

- ✅ **Sim environment install on 22.04 WSL** — every Python dep, all gotchas captured in `SIM_STARTUP.md`.
- ✅ **Agent path imports cleanly** end-to-end (`Frontend`, `Backend`, `ArcPlanner`, `WaypointPlanner`, all reachable).
- ✅ **Phase 0 validation script** (`scripts/phase0_validation.py`): roughness + shadow-casting + SuperPoint feature extraction + multi-distance lookahead + IID statistics + bootstrap CI + per-frame plots. ~900 lines, audited, runs end-to-end in <2 min on RTX 4060.
- ✅ **Phase 0 results captured** for preset 2 (four runs preserved under `output/`):
  - `output/phase0_validation_kp512/` — initial run, 92% saturation, invalidated.
  - `output/phase0_validation_kp2048_no_shadow/` — uncapped, roughness only, H1b ≈ −0.11.
  - `output/phase0_validation_az263_575/` — full ρ_full, sun_az = 263.575° (astropy direct), H4b ≈ +0.05.
  - `output/phase0_validation/` — A/B-test flipped az = 83.575°, H4b ≈ −0.36 (worse → 263.575° is closer to correct).
- ✅ **Findings synthesized** (see `Project_Background.md` Section 13.C in detail). Verdict: predictor is directionally correct (adding shadow flipped H4b from negative to slightly positive), but magnitudes are too small to be useful on preset 2 due to three identified confounds.
- ✅ **Documents up to date**: `Project_Background.md`, `SIM_STARTUP.md`, and this file.

### What's blocked

- ❌ **Sim launch on 22.04**: UE4 4.26 dropped Linux OpenGL → falls back to Vulkan → only llvmpipe (software CPU) → GameThread timeout at 60 s. **Abandoned in favor of 24.04.**
- ✅ **Vulkan on 24.04 RESOLVED** (2026-05-26): added the `ppa:kisak/kisak-mesa` PPA on noble, ran `apt full-upgrade`, Mesa went 25.2 → 26.1.1 and `dzn_icd.x86_64.json` appeared in `/usr/share/vulkan/icd.d/`. `vulkaninfo --summary` now lists `Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)` as `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` / `DRIVER_ID_MESA_DOZEN`. `vkcube` confirmed rendering on the GPU. UE4 should now have a working Vulkan path.
- 🔄 **Full env install in 24.04**: pending. The 24.04 distro currently has Vulkan + apt graphics packages; everything else from `SIM_STARTUP.md` Sections 1.1, 2, 3 has not been replayed there yet.

### What's deliberately not started

- ❌ **Core planner code** — gated on richer Phase 0 results (see "Why we're holding off" below).
- ❌ **Pulling HW3 `dem.py` + `util.py` into `lac/planning/`** — was Phase 1 of the original plan; can be done in parallel with sim debugging if you want a productive task that doesn't depend on Vulkan being fixed.

### Why we're holding off on the planner

Preset 2 has very limited terrain variation (rover trajectory: 7 m × 15 m sub-region, 1.86 m height range, roughness varies only 0.005–0.04 m for 95% of frames). Phase 0 told us:

1. **The predictor concept is plausible** (shadow term flipped H4b sign in the right direction).
2. **The dataset is the wrong test bed** — too little terrain variation, plus three confounds we'd want to control for: (a) rover-cast shadow on its own camera FOV, (b) lookahead under-sampling the camera's mid-distance band, (c) hallucinated single-frame SuperPoint features that don't survive to SLAM matching.

Building the planner on top of weak/inconclusive validation risks 3–5 days of work that we'd have to redo or discard. **The right move is to re-run Phase 0 on a richer trajectory after the sim is up, then build the planner with confirmed validation in hand.**

---

## 3. Inventory: what's in this repo

### Source code

| Path | Status | Description |
|---|---|---|
| `agents/nav_agent.py` | inherited, untouched | The baseline LAC agent (DO NOT MODIFY — we'll fork to `agents/perception_aware_agent.py`) |
| `lac/` | inherited, untouched | The SLAM + perception + planning + mapping modules from Stanford NavLab |
| `lac/planning/waypoint_planner.py` | inherited | Our drop-in replacement target. Signature: `get_waypoint(step, pose) -> (np.ndarray (2,) | None, bool)` |
| `lac/planning/arc_planner.py` | inherited | 41-arc DWA local planner. Hooks for Stretch A (perception-weighted arc selection) |
| `scripts/phase0_validation.py` | **OURS — KEY** | The Phase 0 validation script. ~900 lines. Roughness + shadow + SuperPoint + stats + plots |
| `scripts/train_segmentation.py` | inherited | Where the semantic-class RGB palette is documented (line 21-27) |

### Data

| Path | Status | Description |
|---|---|---|
| `data/DEMs/Moon_Map_01_2_rep0.dat` | **GOT** | LAC preset 2 ground-truth heightmap (180, 180, 4) — channels [x, y, z, rock_bool] |
| `data/DEMs/Moon_Map_01_0_rep0.dat` | got | Preset 0 (no rocks) heightmap, less interesting |
| `data/Example_Implementations/HW3_Final/data/dem_tile.npz` | **GOT** | LRO LOLA 5 m/pixel south-pole DEM tile (10 km × 10 km). Mid-scale eval env |
| `data/Example_Implementations/HW3_Final/data/lac_data/` | **GOT** | 2000 stereo PNGs + `data_log.json` for LAC preset 2. Used by Phase 0 |
| `data/Example_Implementations/HW3_Final/data/LAC/segmentation/` | got | 194 paired (image, semantic) frames for preset 7. Not currently used |
| `data/Example_Implementations/HW3_Final/data/P2_vo_output.npz` | got | Reference VO output from HW3 P2 |
| `data/Example_Implementations/HW3_Final/supplemental/dem.py` | **GOT — REUSE** | Clean DEM class with `query`, `grad`, `slope_deg_grid`, `xy_to_rc`, `downsample`. Pull into `lac/planning/dem.py` (Phase 1) |
| `data/Example_Implementations/HW3_Final/supplemental/util.py` | **GOT — REUSE** | Contains `AStar[T]` base class + `path_length` + `path_max_slope_deg`. Pull `AStar` into `lac/planning/astar.py` |
| `data/Example_Implementations/HW3_Final/AA278_2026_HW3_P4.ipynb` | got | The baseline planner reference notebook |
| `models/unet_v2.pth` | **GOT** | 100 MB UNet++ segmentation weights. Required by `lac/perception/segmentation.py` |

### Documentation

| Path | Description |
|---|---|
| `docs/Project_Background.md` | Technical design doc. Read SECOND (after this file). Section 13 has the latest status |
| `docs/SIM_STARTUP.md` | Install + launch procedure for Ubuntu 24.04 WSL. Read THIRD (when actually setting up) |
| `docs/PROJECT_TURNOVER.md` | **This file**. Status + roadmap. Read FIRST |

### Configs

`configs/config.json` (and variants `five_loops`, `nine_loops`, `spiral`, `triangles`) — JSON config consumed by `nav_agent.py`. See `Project_Background.md` Section 11.5 for the additional fields we'll need.

### Generated outputs (not committed)

| Path | Description |
|---|---|
| `output/phase0_validation_kp512/` | First Phase 0 run, kp cap=512, 92% saturation, invalidated |
| `output/phase0_validation_kp2048_no_shadow/` | Roughness-only baseline, kp cap=2048 |
| `output/phase0_validation_az263_575/` | Full ρ_full at sun_az=263.575° |
| `output/phase0_validation/` | Latest run; A/B-tested az=83.575°, confirms 263.575° is closer to correct |

### Sim (gitignored)

`LunarAutonomyChallenge/LunarAutonomyChallenge/` — 12 GB of UE4 binary + Carla wheels + Leaderboard package. Provided by JHU APL.

---

## 4. Phase 0 findings: full detail

### The hypothesis being tested

> A DEM-derived predictor ρ(x, y) = roughness_normalized × (1 − shadow_mask) is positively correlated with the count of SuperPoint features that the rover's FrontLeft camera will detect when looking ahead at world position (x, y).

This is the predictor that the perception-aware A* will use to bias paths toward visually-rich terrain.

### Data used

- DEM: `data/DEMs/Moon_Map_01_2_rep0.dat` (preset 2, 180 × 180, channels [x, y, z, rock_bool], height range 0.59–2.45 m)
- 2000 stereo frames + ground-truth poses: `data/Example_Implementations/HW3_Final/data/lac_data/`
- Detector: SuperPoint at `max_num_keypoints=2048` (raised from SLAM default 512 after the initial run hit 92% saturation)
- Sun direction: `az=263.575°`, `alt=1.488°` computed via lunarsky/astropy at lunar south pole, date 2023-01-15 00:00:00 (matches `mission_weather.py` initial date)

### Method

For each of the 2000 frames:
1. Compute roughness field (5×5 cell std of z) over the entire 180 × 180 DEM (vectorized, <1 s).
2. Compute shadow mask via ray-cast toward sun (vectorized DDA, <3 s, 47.9% of cells shadowed at this very low sun altitude).
3. Compute ρ_full = roughness × (1 − shadow_mask).
4. At rover XY position AND at lookahead points d ∈ {1.0, 1.5, 2.0, 3.0, 4.0} m along heading, sample ρ_full and roughness.
5. Run SuperPoint on FrontLeft image, count keypoints.
6. Correlate.

Statistics: raw Pearson, IID-subsampled Pearson (frames ≥ 1 m apart, n ≈ 39), Spearman rho, block-bootstrap 95% CI.

### Headline numbers (the best run: `output/phase0_validation_az263_575/`)

| Hypothesis | Pearson_raw | Pearson_iid | Spearman | Interpretation |
|---|---|---|---|---|
| H1a roughness@pose → n_features | **−0.42** | −0.22 | −0.42 | Strong, negative, robust |
| H1b roughness@LA=2m → n_features | −0.11 | −0.19 | −0.16 | Weak negative |
| H2 rocks@LA=2m → n_features | +0.07 | +0.16 | +0.14 | ~zero |
| H4a ρ_full@pose → n_features | −0.41 | −0.16 | −0.41 | Same as H1a (shadow term doesn't add at pose) |
| **H4b ρ_full@LA=3m → n_features** | **+0.05** | **+0.15** | **+0.06** | Weak positive — shadow flipped the sign from H1b |

A/B test for sun-azimuth convention: flipping to `az = 83.575°` produced H4b_pearson_raw ≈ −0.36 at d=3m (worse direction, stronger anti-correlation). So **263.575° is closer to the correct world-frame sun azimuth** than the flipped value, but we haven't done the decisive visual confirmation against rover/rock shadows in actual frame images (requires sim).

### What this means

1. **The predictor framework works in concept**. Adding the shadow mask reliably flipped H4b from clearly negative (−0.15) to slightly positive (+0.05) across all lookahead distances — exactly the direction the design predicts.
2. **Magnitudes are too small to draw planner-design conclusions**. |H4b_iid| < 0.3 by the threshold heuristic — "weak" verdict.
3. **The limiting factors are diagnosable** and addressable:
   - **Preset 2 has insufficient terrain variation.** Rover stays in 7 × 15 m, height range 1.86 m, roughness varies 0.005–0.04 m for 95% of frames. There's no signal to correlate against.
   - **Rover-cast shadow contaminates n_features.** When the rover stands between sun and camera, the camera looks into the rover's own shadow → near-black image → few features → low n_features that has nothing to do with terrain geometry. Example: frame `000034`12.png` (step 3412), n_features=249, predictor says lookahead is lit. Confound, not predictor failure.
   - **Single-point lookahead under-samples the camera FOV.** FrontLeft sees a ground band ~4–10 m ahead; we sample a single cell at a single distance. A forward-cone integration over the actual FOV would be more principled.
   - **n_features includes single-frame hallucinated features** that don't survive to SLAM matching. The truly useful metric is matched/tracked features across consecutive frames using `lac/slam/feature_tracker.py:match_feats`. Two-frame survival via LightGlue is the right Phase 0 v2 response variable.

**Punchline**: do NOT pivot the predictor based on these numbers. Re-run on a richer trajectory with matched-features after the sim is up.

---

## 5. The plan from here (priority order)

### IMMEDIATE

**0. ~~Resolve Vulkan-on-WSL~~ → DONE 2026-05-26.** kisak-mesa PPA on 24.04 + apt full-upgrade gives Mesa 26.1.1 with dzn driver. NVIDIA RTX 4060 visible as `Microsoft Direct3D12 (NVIDIA ...)` `DRIVER_ID_MESA_DOZEN`. `vkcube` renders on the dGPU.

**0a. Replay the full install in the 24.04 distro.** The 24.04 distro currently has Vulkan + apt graphics packages; everything else is fresh. Follow `SIM_STARTUP.md` Sections 1.1 (Conda), 2 (Python deps), 3 (sim install + script edits). Then run `SIM_STARTUP.md` Section 4 to launch the sim.

Success criterion: `./RunLunarSimulator.sh` opens an Unreal window with the lunar scene rendering at >10 fps on the NVIDIA dGPU, AND `./RunLeaderboard.sh` in a second terminal prints `Step: 1, 2, 3, ...` with the rover visibly moving in the sim window.

### CAN RUN IN PARALLEL WITH (0)

**1. Pull `dem.py` into the repo.** Copy `data/Example_Implementations/HW3_Final/supplemental/dem.py` to `lac/planning/dem.py`. Cheap, low-risk, no dependencies on sim being up.

**2. Pull `AStar` into the repo.** Copy the `AStar[T]` class (and `path_length`, `path_max_slope_deg`) from `data/Example_Implementations/HW3_Final/supplemental/util.py` to `lac/planning/astar.py`. Drop the unused `LoopClosureMeasurement` dataclass.

### POST-SIM (gated on Phase 0)

**3. Run baseline `nav_agent.py` end-to-end on one richer preset.** Capture the ground-truth heightmap from `results/Moon_Map_01_<PRESET>_rep0.dat`. Confirm SLAM RMSE measurement works end-to-end. Optionally pick presets 5 / 7 / 9 to compare.

**4. Visually verify sun azimuth convention.** Open `dem_overlays.png` from a Phase 0 run; open a corresponding actual frame image; confirm rock shadows fall in the same world-frame direction. If not, sweep `SUN_AZIMUTH_DEG` over {83.575, 96.425, 186.425, 0, 90, 180, 270}.

**5. Plan and capture a rich trajectory.** Drive the rover manually (via `human_agent.py` or by editing `MISSIONS_SUBSET` to a preset with varied terrain) through a path that deliberately crosses flat regions AND rough/elevated regions. Save the resulting `data_log.json` + image frames.

**6. Re-run Phase 0 with the rich trajectory + matched-features as the response variable.** Modify `scripts/phase0_validation.py` to also extract features from frame i+1 and call `lac/slam/feature_tracker.py:match_feats(feats_i, feats_{i+1}, min_score=0.5)`. Use `n_survived = len(matches)` as the response variable alongside (or instead of) `n_features`. Re-correlate vs ρ_full. The decisive test of whether to build the planner.

**Verdict thresholds** (using H4b Pearson_iid on the richer dataset with matched-features):
- `|r| ≥ 0.4` → STRONG. Build the planner as designed.
- `0.2 ≤ |r| < 0.4` → MODERATE. Build the planner, plan for Stretch B online refinement to compensate.
- `|r| < 0.2` → WEAK. Pivot the predictor (consider learned ρ, or shadow-only ρ without roughness).

### CORE PLANNER (gated on (6))

**7. Build `lac/planning/perception_map.py`.** A class that wraps a DEM and exposes:
- `roughness_field` (180, 180) — computed at construction
- `shadow_mask(sun_az_deg, sun_alt_deg)` (180, 180) bool — computed via the algorithm already in `scripts/phase0_validation.py:compute_shadow_mask`
- `rho_full(sun_az_deg, sun_alt_deg)` (180, 180) — the multiplicative product
- `get_uncertainty_cost(x, y) -> float` — returns `1 − ρ_full(x, y)` for use as A* edge cost. Handles out-of-bounds.

**8. Build `lac/planning/ekf_covariance.py`.** Simplified 2D `[x, y]` state EKF that propagates predicted localization uncertainty along a candidate path. Returns `log det Λ_N` (D-optimality) as a scalar path score. Used for offline path ranking, not for planning per se. See `Project_Background.md` Section 8.2.

**9. Build `lac/planning/perception_aware_planner.py`.**
- Subclass `AStar[GridCoord]` (from step 2).
- `edge_cost(n1, n2) = dist * (1 + α * (slope(n2)/θ_ref)^p + β * (1 - ρ(n2)))`. Multiplicative form matches HW3 P4 baseline plus our additive ρ term.
- `heuristic`: Euclidean (admissible because the multiplier is always ≥ 1).
- Expose `get_waypoint(step, pose) → (np.ndarray (2,) | None, bool)` to match `WaypointPlanner` exactly (drop-in interface).
- Sub-goal handling for goal-to-goal: input is a single goal pose; planner generates a sequence of waypoints from start to goal by sampling the A* path at ~2 m intervals.

**10. Build `agents/perception_aware_agent.py`.** Drop-in fork of `agents/nav_agent.py` — only `setup()` changes (instantiate `PerceptionAwarePlanner` instead of `WaypointPlanner`).

### EXPERIMENTS

**11. LRO mid-scale experiments.** Plan paths between hand-picked start/goal pairs on the LRO 5 m/pixel tile. Compare slope-aware A* baseline (β=0) vs perception-aware (β > 0). Metrics: predicted D-optimality (from step 8), path length, max slope, ρ-integral. **Visual paper figures.**

**12. LAC small-scale experiments.** Pick 3–5 hand-picked (start, goal) pairs on Moon_Map_01 presets with varied terrain. Run both planners through the actual sim. Measure SLAM RMSE via `positions_rmse_from_poses(slam_poses, gt_poses)` (already in `nav_agent.finalize()`). **Quantitative paper claims.**

**13. Ablations.** β sweep on both environments. Shadow on/off. (Stretch A/B/C only if time permits.)

### PAPER (overlapping experiments)

**14. ION-format LaTeX, 6–8 pages.** Section structure already in `Project_Background.md` Section 11.

---

## 6. Critical "do not"s

1. **Do not rebuild the predictor based on preset 2 alone.** It's directionally correct but underpowered. The plan is to re-validate on richer data.
2. **Do not modify `lac/slam/frontend.py` or `lac/slam/backend.py`** — the SLAM stack is inherited as-is. Any changes break parity with the rest of the team's work.
3. **Do not modify `agents/nav_agent.py`** — fork to `agents/perception_aware_agent.py`.
4. **Do not modify `lac/planning/arc_planner.py`** except for the single Stretch A perception term (and only after the core planner is working).
5. **Do not introduce a yaw sign flip.** Positive yaw in the simulator API is **clockwise** (opposite to standard math convention). The existing code already handles this.
6. **Do not use `MAP_SIZE` (180) as the scene size in meters.** Scene is ±20 m, map is ±13.5 m (180 cells × 0.15 m/cell). These are different numbers.
7. **Do not output velocity commands from the global planner.** Output 2-D `[x, y]` waypoints only — `ArcPlanner` produces velocity commands.
8. **Do not run `pip install --force-reinstall -r LunarAutonomyChallenge/.../requirements.txt`** without then re-installing our pins on top. The sim's requirements would downgrade numpy and matplotlib.
9. **Do not skip `PYTHONNOUSERSITE=1`.** User-site shadowing has eaten hours of debugging time.
10. **Do not implement Stretch B or C** before the core planner is running and validated end-to-end.
11. **Do not implement MPC for loop closure incentive.** A* greedy γ term is the chosen approach. MPC is documented in Section 8.7 as a design decision for the paper — not as future implementation work.

---

## 7. Open decisions you may need to make

1. **If Vulkan-on-WSL stays broken**: move to Durand 339 lab workstation (yes/no, when).
2. **After step 4 visual verification**: lock in the world-frame sun azimuth convention. Document the chosen value in `Project_Background.md` and the predictor code.
3. **After step 6**: STRONG/MODERATE/WEAK call. Determines whether the planner is built as designed or pivoted.
4. **Around day 7 of 10**: cut decision for Stretch A/B/C. Default is to skip; only add if core + experiments are done by then.

---

## 8. Contact / external resources

- **TA Adam Dai** (Stanford NavLab): primary contact for the LAC sim + DEMs. Delivered preset 2 DEM directly. Will deliver others on request.
- **JHU APL** (challenge organizer): owns `lunar-autonomy-challenge.jhuapl.edu/Challenge-Documentation/`. Provides the sim ZIP, the UNet weights, and the documentation.
- **Project repo**: `~/Stanford/AA278/Lunar_Perception_Aware_Planning` on the user's WSL machine.

---

## 9. For a coding agent starting fresh in the Ubuntu 24.04 distro

If you're a Claude Code agent that just got handed this project after a fresh WSL install, the linear path to get back to the current state is:

1. Read this entire document (`docs/PROJECT_TURNOVER.md`).
2. Read `docs/Project_Background.md` for the technical design, especially Section 7 (existing planner / SLAM interfaces), Section 8 (what to build), and Section 13 (current status).
3. Read `docs/SIM_STARTUP.md` and execute it step by step. The "Quick sanity-check command sequence" at the end is your gate — when all 8 items return `present` / OK, you're caught up.
4. If you're still blocked on Vulkan-on-WSL at Section 1.3 of SIM_STARTUP.md after ~30 minutes of trying, surface the blocker and ask whether to escape to the Durand 339 lab workstation. Do not burn more than 1 hour on Vulkan debugging without checking in.
5. Once the sim runs, run `python scripts/phase0_validation.py` to confirm the preset 2 Phase 0 still reproduces the documented numbers (sanity check).
6. Proceed with Section 5 of this file (the priority-ordered plan).

Do NOT start building the planner code until Phase 0 has been re-run on a richer trajectory with matched-features per Section 5 step 6 of this file. The temptation will be there; resist it. Three days of planner work on the wrong predictor is much worse than one day of Phase 0 v2 work.

When you finish a meaningful chunk of work, **update this file** under "Where we left off" so the next handoff has a clean starting point.
