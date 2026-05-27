# Project Turnover — Lunar Perception-Aware Planning

> **Purpose**: This document is the handoff packet for any AI agent or human picking up this project mid-stream. Read this first. Then read `Project_Background.md` (technical design) and `SIM_STARTUP.md` (install procedure). The three files together fully specify the project's intent, status, work product, and immediate next steps.

---

## 1. Project at a glance

**Course**: AA278 Lunar PNT, Stanford, Spring 2026. Solo final project by Santiago Thorup.
**Deadlines**: slides 2026-06-01 (in-class), report 2026-06-05 (4 PM PT, ION format, 6–8 pages).
**Status as of 2026-05-26**: Phase 0 validation complete. Python environment on WSL2 fully installed. **LAC simulator on WSL2 definitively blocked** — UE4 4.26 hangs due to a D3D12 fence timeout in Mesa's `dzn` driver (see `Project_Background.md` Section 13.D for full investigation). **Resolution: run the sim on an SSH server with native Linux + NVIDIA GPU using `Xvfb` for headless rendering.** Server setup is the immediate next step (see Section 9 of this file). No planner code written yet — deliberately gated on richer Phase 0 results from the sim.

**One-sentence project description**: Build a **goal-to-goal** trajectory planner for a lunar rover that proactively routes through visually feature-rich terrain to minimize SLAM localization error, combining DEM-derived roughness + sun shadow casting into a feature-density predictor used as a soft cost in A*.

**Baseline comparison**: Slope-aware A* from AA278 HW3 Problem 4 (the implementation is at `data/Example_Implementations/HW3_Final/AA278_2026_HW3_P4.ipynb` and `data/Example_Implementations/HW3_Final/supplemental/dem.py` + `supplemental/util.py`). The perception-aware planner extends P4's cost function with an additive ρ-weighted term.

**Primary metric**: SLAM RMSE vs. ground truth over a planned path (`positions_rmse_from_poses` already in `lac/util.py`).

**Two evaluation environments**:
1. **Mid-scale (LRO real lunar tile)**: `data/Example_Implementations/HW3_Final/data/dem_tile.npz` (2000 × 2000 cells at 5 m/pixel = 10 km × 10 km, real LRO LOLA south-pole DEM). No SLAM here — algorithmic + visual demo of paths on actual lunar terrain.
2. **Small-scale (LAC simulator)**: 27 m × 27 m simulator environment (Moon_Map_01) with ~10 preset variations. Hand-picked (start, goal) pairs. RMSE measured end-to-end.

---

## 2. Where we left off (read this carefully)

### What's done

- ✅ **Python env on Ubuntu 24.04 WSL2 — COMPLETE**: conda `lac` env, Python 3.10.20, `PYTHONNOUSERSITE=1`, PyTorch 2.4.1+cu121, LightGlue, apriltag, Carla 0.9.15, and all deps. Agent-path imports (`Frontend`, `Backend`, `ArcPlanner`, `WaypointPlanner`) confirmed green. See `SIM_STARTUP.md` Section 2 for the install procedure.
- ✅ **Phase 0 validation script** (`scripts/phase0_validation.py`): roughness + shadow-casting + SuperPoint feature extraction + multi-distance lookahead + IID statistics + bootstrap CI + per-frame plots. ~900 lines, audited, runs end-to-end in <2 min on RTX 4060.
- ✅ **Phase 0 results captured** for preset 2 (four runs preserved under `output/`):
  - `output/phase0_validation_kp512/` — initial run, 92% saturation, invalidated.
  - `output/phase0_validation_kp2048_no_shadow/` — uncapped, roughness only, H1b ≈ −0.11.
  - `output/phase0_validation_az263_575/` — full ρ_full, sun_az = 263.575° (astropy direct), H4b ≈ +0.05.
  - `output/phase0_validation/` — A/B-test flipped az = 83.575°, H4b ≈ −0.36 (worse → 263.575° is closer to correct).
- ✅ **Findings synthesized** (see `Project_Background.md` Section 13.C). Predictor directionally correct; magnitudes too small on preset 2 due to limited terrain variation — re-run on richer trajectory required.
- ✅ **WSL2 sim investigation complete**: dzn (Mesa D3D12 Vulkan) gets UE4 to accept the device but hangs at D3D12 PSO init (GPU fence never signals). Root cause: Mesa `dzn` bug with UE4 4.26 workload. Compat layer written at `tools/lac_vulkan_compat_layer.c` (archived for reference).
- ✅ **Documents up to date**: all three docs reflect current state as of 2026-05-26.

### What's blocked / in progress

- ❌ **LAC simulator on WSL2 — CLOSED**: D3D12 fence timeout inside Mesa `dzn`. Not fixable externally. Full write-up in `Project_Background.md` Section 13.D.
- 🔄 **SSH server setup — NEXT STEP**: server identified, setup instructions in Section 9 of this file.

### What's deliberately not started

- ❌ **Core planner code** — gated on Phase 0 re-run confirming |r| ≥ 0.2 on richer trajectory (see "Why we're holding off" below).
- ❌ **Pulling HW3 `dem.py` + `util.py` into `lac/planning/`** — Phase 1 of the original plan; can (and should) be done on the WSL2 machine in parallel with server setup.

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

`LAC_SIM/` — 12 GB of UE4 binary + Carla wheels + Leaderboard package. Provided by JHU APL.

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

**0. Set up the SSH server for headless sim runs.** WSL2 cannot run UE4 4.26 (dzn D3D12 fence hang, definitively closed). The sim runs headlessly on a native Linux + NVIDIA GPU server via `Xvfb`. Full step-by-step instructions in **Section 9 of this file**. This is the blocking dependency for all sim-dependent work.

Success criterion: `./RunLeaderboard.sh` on the server prints `Step: 1, 2, 3, ...` and a `results/Moon_Map_01_<PRESET>_rep0.dat` file appears after the run completes.

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
8. **Do not run `pip install --force-reinstall -r LAC_SIM/requirements.txt`** without then re-installing our pins on top. The sim's requirements would downgrade numpy and matplotlib.
9. **Do not skip `PYTHONNOUSERSITE=1`.** User-site shadowing has eaten hours of debugging time.
10. **Do not implement Stretch B or C** before the core planner is running and validated end-to-end.
11. **Do not implement MPC for loop closure incentive.** A* greedy γ term is the chosen approach. MPC is documented in Section 8.7 as a design decision for the paper — not as future implementation work.

---

## 7. Open decisions you may need to make

1. ~~**If Vulkan-on-WSL stays broken**~~ **→ DECIDED 2026-05-26**: WSL2 cannot run the sim. Using SSH server with native Linux + NVIDIA GPU and `Xvfb` headless rendering.
2. **After step 4 visual verification**: lock in the world-frame sun azimuth convention. Document the chosen value in `Project_Background.md` and the predictor code.
3. **After step 6**: STRONG/MODERATE/WEAK call. Determines whether the planner is built as designed or pivoted.
4. **Around day 7 of 10**: cut decision for Stretch A/B/C. Default is to skip; only add if core + experiments are done by then.

---

## 8. Contact / external resources

- **TA Adam Dai** (Stanford NavLab): primary contact for the LAC sim + DEMs. Delivered preset 2 DEM directly. Will deliver others on request.
- **JHU APL** (challenge organizer): owns `lunar-autonomy-challenge.jhuapl.edu/Challenge-Documentation/`. Provides the sim ZIP, the UNet weights, and the documentation.
- **Project repo**: `~/Stanford/AA278/Lunar_Perception_Aware_Planning` on the user's WSL machine.

---

## 9. For a coding agent on the SSH server — complete setup playbook

This section is the complete, step-by-step guide for getting the experimental pipeline operational on a **native Linux SSH server with an NVIDIA GPU**. Follow it in order. Do not skip steps.

> **Context**: The LAC simulator (UE4 4.26 + Carla) cannot run on WSL2 due to a Mesa `dzn` D3D12 driver bug. It runs correctly on native Linux with a native NVIDIA Vulkan driver. Sensor cameras in Carla render to GPU textures independently of any display window, so headless operation (`Xvfb`) produces full-fidelity images, SLAM output, and RMSE measurements identical to an interactive run.

---

### Step 0 — Read first

1. Read this entire document.
2. Read `docs/Project_Background.md`, especially Sections 7 (SLAM/planner interfaces), 8 (what to build), 13.A (evaluation plan), 13.C (Phase 0 findings), and 13.D (WSL2 investigation — explains why we're here).
3. Skim `docs/SIM_STARTUP.md` Section 0 (Native Linux path) — it is the source of truth for the install procedure. The sections that follow it cover WSL2 and are now legacy.

---

### Step 1 — Verify server hardware and OS

```bash
nvidia-smi                     # must show an NVIDIA GPU with driver installed
uname -r                       # Linux kernel
lsb_release -a                 # Ubuntu 20.04 / 22.04 / 24.04 all work
df -h ~                        # need ≥ 20 GB free (sim is ~12 GB + env ~4 GB)
python3 --version              # any 3.x — we'll install our own via conda
```

**Required**: NVIDIA GPU with ≥ 4 GB VRAM (8 GB recommended), NVIDIA proprietary driver installed, Ubuntu 20.04–24.04.

---

### Step 2 — Install apt dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential pkg-config git \
    xvfb \
    libgl1-mesa-glx libglib2.0-0 \
    cmake
```

- `xvfb` — virtual framebuffer for headless UE4 rendering (essential)
- `libgl1-mesa-glx` — OpenGL runtime (needed by some Python packages)
- `cmake` — needed to build `apriltag` from source

---

### Step 3 — Install Miniconda and create the `lac` env

```bash
cd /tmp
curl -sLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash   # reload shell

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -y -n lac python=3.10
conda activate lac

# CRITICAL: prevent user-site shadowing — without this, pip silently ignores
# version pins because ~/.local/lib/python3.10/ takes precedence.
conda env config vars set PYTHONNOUSERSITE=1 -n lac
conda deactivate && conda activate lac
```

---

### Step 4 — Clone / transfer the repo

**Option A — git clone** (if the repo is on GitHub):
```bash
mkdir -p ~/Stanford/AA278 && cd ~/Stanford/AA278
git clone <REPO_URL> Lunar_Perception_Aware_Planning
cd Lunar_Perception_Aware_Planning
```

**Option B — scp from WSL2 machine** (if not yet pushed):
```bash
# Run on your laptop (WSL2 terminal):
scp -r ~/Stanford/AA278/Lunar_Perception_Aware_Planning \
    user@server:~/Stanford/AA278/
```

Set `REPO` for the rest of these instructions:
```bash
export REPO="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning"
```

---

### Step 5 — Install Python dependencies

```bash
conda activate lac
cd $REPO

# 1. Pin cmake < 4 (apriltag 0.0.16's CMakeLists rejects cmake 4.x syntax)
pip install "cmake<4"

# 2. Install apriltag before requirements.txt (needs --no-build-isolation to
#    find our pinned cmake; a failing apriltag build aborts the whole transaction)
pip install --no-build-isolation apriltag==0.0.16

# 3. Install PyTorch with CUDA 12.1 index (requirements.txt doesn't specify the
#    index URL; installing first prevents it being overridden by the CPU wheel)
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Remaining deps
pip install -r requirements.txt
pip install -e .

# 5. Four undocumented deps used by the agent path
pip install imageio munch segmentation-models-pytorch opt_einsum
```

**6. LightGlue** (needed by SLAM feature tracking):
```bash
mkdir -p ~/opt && cd ~/opt
git clone https://github.com/cvg/LightGlue.git
cd LightGlue && pip install -e .
```

**Verify imports**:
```bash
conda activate lac
python -c "
import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())
import carla; print('carla OK')
" 2>&1 | grep -v Warning
```

Expected: `torch 2.4.1+cu121  cuda= True` and `carla OK`.

---

### Step 6 — Transfer the LAC simulator (`LAC_SIM/`)

The sim is a ~12 GB folder that is **gitignored** — it is not in the repo. Transfer it from the WSL2 machine or download from JHU APL:

**Option A — scp from WSL2 laptop**:
```bash
# Run on your laptop (WSL2 terminal) — takes ~20 min on a fast connection:
scp -r ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM \
    user@server:$REPO/
```

**Option B — copy from zip** (if you have the original JHU APL zip on the server):
```bash
cd $REPO
unzip /path/to/LunarAutonomyChallenge.zip
mv LunarAutonomyChallenge LAC_SIM
```

After transfer, confirm layout:
```bash
ls $REPO/LAC_SIM/
# Must show: RunLunarSimulator.sh  RunLeaderboard.sh  LunarSimulator/  Leaderboard/  wheelhouse/
```

---

### Step 7 — Install sim Python dependencies

```bash
conda activate lac
SIM="$REPO/LAC_SIM"

pip install "$SIM/wheelhouse/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl"
pip install dictor==0.1.12 tabulate==0.9.0 pygame==2.5.2

python -c "import carla, dictor, tabulate, pygame; print('sim deps OK')"
```

---

### Step 8 — Transfer model weights and data files

These are gitignored and must be copied manually:

```bash
# UNet segmentation weights (~100 MB) — required by lac/perception/segmentation.py
mkdir -p $REPO/models
scp user@laptop:$REPO/models/unet_v2.pth $REPO/models/
# OR download from the JHU APL portal under "Model weights"

# Preset 2 DEM (needed for Phase 0 re-run)
mkdir -p $REPO/data/DEMs
scp user@laptop:$REPO/data/DEMs/Moon_Map_01_2_rep0.dat $REPO/data/DEMs/

# Preset 2 image frames + ground-truth poses (needed for Phase 0)
scp -r user@laptop:$REPO/data/Example_Implementations/HW3_Final/data/lac_data \
    $REPO/data/Example_Implementations/HW3_Final/data/

# LRO DEM tile (needed for mid-scale experiments)
scp user@laptop:$REPO/data/Example_Implementations/HW3_Final/data/dem_tile.npz \
    $REPO/data/Example_Implementations/HW3_Final/data/
```

---

### Step 9 — Configure `RunLunarSimulator.sh` for native Linux

On native Linux with the NVIDIA proprietary driver, no WSL2-specific env vars are needed. Replace the file contents:

```bash
sudo tee $REPO/LAC_SIM/RunLunarSimulator.sh << 'EOF'
#!/bin/bash
# Native Linux + NVIDIA GPU launch wrapper.
# No WSL2/D3D12 env vars needed — native NVIDIA Vulkan driver used directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"
bash "$SIMULATOR_ROOT/LAC.sh" "$@"
EOF
```

*(If you don't have sudo, the file may not be root-owned on the server — try `chmod` and a direct edit first.)*

---

### Step 10 — Configure `RunLeaderboard.sh` with server paths

Edit the `TEAM_CODE_ROOT` line in `RunLeaderboard.sh` to point to where you cloned the repo on the server:

```bash
# Check what it currently says:
grep TEAM_CODE_ROOT $REPO/LAC_SIM/RunLeaderboard.sh

# If it still points to the WSL2 path (/home/sthorup/...), update it:
sed -i "s|export TEAM_CODE_ROOT=.*|export TEAM_CODE_ROOT=\"$REPO\"|" \
    $REPO/LAC_SIM/RunLeaderboard.sh

# Verify:
grep TEAM_CODE_ROOT $REPO/LAC_SIM/RunLeaderboard.sh
```

---

### Step 11 — Make scripts executable

```bash
SIM="$REPO/LAC_SIM"
chmod +x "$SIM/RunLunarSimulator.sh" "$SIM/RunLeaderboard.sh" \
         "$SIM/LunarSimulator/LAC.sh" \
         "$SIM/LunarSimulator/LAC/Binaries/Linux/LAC-Linux-Shipping"
mkdir -p "$SIM/results"
```

---

### Step 12 — Test headless sim launch

The sim runs headlessly using `Xvfb` (virtual framebuffer). Sensor cameras render to GPU textures regardless of the display — all image data is produced identically to an interactive run.

**Terminal A — start virtual display and launch sim**:
```bash
# Start the virtual display (do this once per server session)
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

conda activate lac
cd $REPO/LAC_SIM
./RunLunarSimulator.sh
```

Wait for the UE4 engine to finish loading — this takes **60–120 seconds** on first run (shader cache warm-up). The terminal will print a UE4 version string and then go quiet. This is normal — the sim is waiting for a Carla client to connect.

**Terminal B — launch the leaderboard/agent** (after Terminal A has gone quiet):
```bash
export DISPLAY=:99
conda activate lac
cd $REPO/LAC_SIM
./RunLeaderboard.sh
```

**Success criterion**: Terminal B prints `Step: 1`, `Step: 2`, ... The mission runs and completes. A file `$REPO/LAC_SIM/results/Moon_Map_01_<PRESET>_rep0.dat` is written.

**If you want to monitor visually** (optional): install a VNC server and connect from your laptop:
```bash
sudo apt install -y tigervnc-standalone-server
vncserver :1 -geometry 1920x1080 -depth 24
# Then: DISPLAY=:1 ./RunLunarSimulator.sh
# Connect your VNC viewer to server-ip:5901
```

---

### Step 13 — Run the full sanity check

```bash
conda activate lac
cd $REPO
echo "Python    : $(python --version)"
echo "Torch     : $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())')"
echo "Carla     : $(python -c 'import carla; print("0.9.15 OK")')"
echo "CUDA GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Sim dir   : $(test -d $REPO/LAC_SIM && echo present || echo MISSING)"
echo "DEM       : $(test -f $REPO/data/DEMs/Moon_Map_01_2_rep0.dat && echo present || echo MISSING)"
echo "UNet      : $(test -f $REPO/models/unet_v2.pth && echo present || echo MISSING)"
echo "lac_data  : $(test -d $REPO/data/Example_Implementations/HW3_Final/data/lac_data && echo present || echo MISSING)"
```

All items should return `present` / OK. `cuda= True` is required.

---

### Step 14 — Run Phase 0 sanity check

Confirms the analysis code still works on the server with the same data:

```bash
conda activate lac
cd $REPO
python scripts/phase0_validation.py
```

Expected: completes without error in <5 minutes, produces plots under `output/phase0_validation_*/`. The headline numbers should match the values in Section 4 of this file (H4b Pearson_raw ≈ +0.05 at d=3m). This is a no-new-data sanity check — it uses the existing preset 2 frames already in `data/`.

---

### Step 15 — Proceed with the experimental pipeline

Once Steps 12–14 pass, you're fully operational. Follow the priority-ordered plan in **Section 5** of this document:

1. **Run baseline `nav_agent.py` on a richer preset** (presets 5, 7, or 9 recommended — more terrain variation than preset 2). Collect ground-truth heightmap from `results/`.
2. **Visually verify sun azimuth** — compare `dem_overlays.png` from Phase 0 output against actual frame images; confirm shadow direction matches.
3. **Re-run Phase 0** with the richer trajectory + matched-features (LightGlue) as the response variable. Decision threshold: |H4b r| ≥ 0.2 on IID-subsampled data.
4. **Build the planner** (Phases 1–5 in Section 5) — only after Phase 0 confirms the predictor.
5. **Run experiments** (Phases 6–7 in Section 5).

**Do NOT start building the planner until Phase 0 re-run confirms |r| ≥ 0.2.** See Section 5 for the full rationale.

---

### Common server-specific failures

| Symptom | Cause | Fix |
|---|---|---|
| `./RunLunarSimulator.sh` hangs, no output | `DISPLAY` not set | `export DISPLAY=:99` before running; confirm `Xvfb :99` is running (`ps aux \| grep Xvfb`) |
| Sim starts but `RunLeaderboard.sh` fails with `ModuleNotFoundError: leaderboard` | PYTHONPATH not set | Always run via `./RunLeaderboard.sh` which sets PYTHONPATH; do not call `nav_agent.py` directly |
| UE4 crashes with "SIGSEGV" in Intel D3D12 libs | Intel iGPU D3D12 driver crash during cleanup | Harmless on native Linux if there's no Intel iGPU; if there is one, add `export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` to `RunLunarSimulator.sh` |
| `torch.cuda.is_available()` returns False | PyTorch CPU wheel installed | `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121` |
| `import torch` returns wrong version | User-site shadowing | `conda env config vars set PYTHONNOUSERSITE=1 -n lac` then reactivate |
| `./RunLeaderboard.sh` finds wrong `TEAM_CODE_ROOT` | Path still points to WSL2 | Re-run Step 10 with server's actual repo path |
| UE4 exits after ~14 s with no output | First-run pipeline issue or missing display | Ensure `DISPLAY=:99` is set and Xvfb is running; check `~/.config/Epic/LAC/Saved/Logs/LAC.log` if it exists |

---

When you finish a meaningful chunk of work, **update the "Where we left off" section** (Section 2) of this file so the next handoff is clean.
