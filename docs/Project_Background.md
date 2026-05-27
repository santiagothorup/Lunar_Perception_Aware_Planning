# Project Background: Perception-Aware Path Planning for Lunar Rover Autonomy

> **Purpose of this file:** This document is the complete context briefing for a Claude Code agent assisting with implementation, experimentation, and paper writing for Santiago Thorup's AA278 final project at Stanford. Read this entire file before taking any action. All implementation decisions should be consistent with the technical design described here.

---

## 1. Course and Project Overview

**Course:** AA278 — Lunar Positioning, Navigation, and Timing (PNT), Stanford University  
**Project type:** Solo final project  
**Student:** Santiago Thorup

**One-sentence project description:**  
Design and implement a perception-aware global path planner for a lunar rover that proactively routes through visually feature-rich terrain to maintain high-quality SLAM localization — combining a DEM-derived feature density predictor, online map refinement, perception-weighted local arc selection, and loop closure incentivization.

**Core research insight / positioning:**  
Existing lunar SLAM work (Dai et al. 2026, ShadowNav JPL 2024) improves localization *reactively* — better feature extraction given whatever images the rover encounters. This project plans paths so the rover *encounters better images to begin with*. No prior work combines prior DEM geometry, predicted illumination shadowing, and Fisher information path planning for proactive lunar localization quality.

---

## 2. Deliverables and Deadlines

| Deliverable | Deadline | Notes |
|---|---|---|
| Final presentation slides | June 1, 2026 (before 2:45 PM PT) | Slides submitted to Gradescope must match presented slides exactly |
| Final presentation | June 1 or 3, 2026 | In-class |
| Final project report | June 5, 2026 (4:00 PM PT) | ION format, single-column, 6–8 pages (solo), excluding references/appendices |

**Report format:** ION GNSS+ conference style. See: https://www.ion.org/gnss/author-resource-center-preparation.cfm  
**Code:** All implementation must be in a public Git repository. Links must appear in the paper.

---

## 3. Project Scope — Two-Tier Structure

The project is deliberately scoped in two tiers so it is robust to time pressure:

**Part 1 — Core contribution (must complete):**  
A perception-aware global planner that demonstrably reduces predicted localization uncertainty versus a shortest-path baseline, evaluated in simulation under realistic SLAM noise. This alone constitutes a complete, defensible ION paper.

**Part 2 — Stretch goals (build in order, stop when time runs out):**

- **Stretch A — Perception-Aware Local Arc Selection (Option B):** Add a ρ-weighted perception term to `ArcPlanner`'s arc scoring so the local planner also prefers feature-rich terrain when multiple arcs are otherwise comparable. ~2 hours of work; a 10-line augmentation to `arc_planner.py`. Provides a clean ablation: global perception planning alone vs. global + local.

- **Stretch B — Online ρ Refinement (Option A):** At each `update_map()` call, update a running empirical ρ estimate from observed SLAM feature counts and replan A* from the current position using the blended map. Closes the perception-planning loop: prior DEM → observed refinement → replanning.

- **Stretch C — Loop Closure Incentivization:** Add a third term to the planning objective that routes the rover back through previously mapped, feature-rich locations to proactively trigger SLAM loop closures and correct accumulated drift. See Section 8.5 for full design.

The MPC stretch goal from the original proposal has been **deliberately dropped**. The existing `ArcPlanner` already handles dynamic feasibility locally (Dubins arcs, curvature constraints, rock avoidance). Since the global planner outputs waypoints that ArcPlanner steers toward — not a trajectory the rover follows directly — trajectory smoothness at the global level does not affect execution behavior. MPC would add significant implementation complexity for no measurable improvement. See Section 8.6 for the full MPC vs. A* trade-off analysis.

---

## 4. Codebase: Stanford NavLab Lunar Autonomy Challenge Repo

**Repository:** https://github.com/Stanford-NavLab/lunar_autonomy_challenge  
**Starting point:** Fork the repo. All new code lives in the fork.  
**Key instruction from TA Adam Dai:** Use the existing `nav_agent.py` and `lac/` module as the foundation. Do not build a full SLAM stack from scratch. Focus implementation effort purely on the perception-aware planning layer.

### 4.1 Repository Structure

Only the files most relevant to this project are listed below. Many supporting modules exist (perception, localization, controllers, GTSAM factor utilities, third-party RAFT stereo, rerun logging, etc.) but are not directly touched by the perception-aware planning layer.

```
lunar_autonomy_challenge/
├── agents/
│   ├── nav_agent.py            ← Main agent entry point (DO NOT MODIFY — fork instead)
│   ├── data_collection_agent.py
│   └── visualization_agent.py
├── lac/
│   ├── params.py               ← All shared constants (critical reference)
│   ├── util.py                 ← Misc helpers (transform_to_numpy, RMSE, etc.)
│   ├── planning/
│   │   ├── waypoint_planner.py     ← Global waypoint sequencer (YOUR REPLACEMENT TARGET)
│   │   ├── arc_planner.py          ← Local DWA arc selector (DO NOT MODIFY except Stretch A)
│   │   ├── temporal_arc_planner.py ← Time-aware variant of ArcPlanner (not used by nav_agent)
│   │   ├── collision_recovery.py   ← Backup-maneuver helpers
│   │   └── waypoint_generation.py  ← Pre-baked coverage patterns (reference only)
│   ├── slam/
│   │   ├── frontend.py             ← Stereo VO + segmentation + rock detection (DO NOT MODIFY)
│   │   ├── backend.py              ← Pose graph SLAM with loop closure (DO NOT MODIFY)
│   │   ├── semantic_feature_tracker.py, feature_tracker.py, visual_odometry.py
│   │   ├── loop_closure.py         ← keyframe_estimate_loop_closure_pose
│   │   ├── gtsam_util.py, gtsam_factor_graph.py
│   │   ├── rock_tracker.py, slam.py
│   ├── mapping/
│   │   ├── mapper.py               ← Mapper class + process_map()
│   │   ├── map_utils.py            ← get_geometric_score, get_rocks_score
│   │   ├── interpolation.py, terrain_map.py
│   ├── control/
│   │   ├── dynamics.py             ← Dubins trajectory model used by ArcPlanner
│   │   ├── controller.py, steering.py
│   ├── localization/               ← EKF, factor graphs, IMU recovery (not invoked by NavAgent)
│   ├── perception/                 ← Segmentation, depth, PnP, vision utilities
│   └── utils/
│       ├── frames.py               ← Transform helpers (invert_transform_mat, apply_transform, …)
│       ├── camera.py, geometry.py, plotting.py
│       ├── dashboard.py, visualization.py
│       ├── data_logger.py, rerun_interface.py
├── configs/                        ← config.json, five_loops.json, nine_loops.json, spiral.json,
│                                     triangles.json
├── docs/                           ← geometry.json (rover/lander geometry), missions.xml,
│                                     Dockerfile  (NOTE: no frame_sign_conventions.md exists in the
│                                     repo — frame conventions are documented in this file and in
│                                     comments inside lac/utils/frames.py)
├── scripts/                        ← Offline evaluation / training scripts
├── thirdparty/raft_stereo/         ← Vendored stereo network
└── data/                           ← Heightmaps, DEMs go here (gitignored, not yet present)
```

### 4.2 Files You Will Create (New Code)

All new files live under `lac/planning/` and a new agent file:

```
lac/planning/
├── perception_map.py              ← Feature density predictor (ρ map from DEM + sun angle)
├── perception_aware_planner.py    ← Perception-aware A* global planner (core + Stretch B + C)
├── ekf_covariance.py              ← EKF covariance propagation along a path
└── loop_closure_map.py            ← Loop closure opportunity map (Stretch C)

agents/
└── perception_aware_agent.py      ← Thin fork of nav_agent.py (minimal changes)
```

**Note on `arc_planner.py`:** Stretch A adds a small perception term to arc scoring. This is the one exception to the "do not modify" rule — the change is additive (a soft γ term), does not alter collision avoidance logic, and is guarded by a flag so it can be disabled for ablation.

---

## 5. Coordinate Frames and Conventions (CRITICAL — read carefully)

Source of truth: this section + comments in `lac/utils/frames.py`. (There is **no** `docs/frame_sign_conventions.md` file in the repo despite the name appearing in some prior planning notes.)

**Global frame (world frame):**
- Right-handed coordinate system
- `+x` = forward, `+y` = left, `+z` = up
- Origin is at the lander position at mission start

**Robot/body frame:**
- Same axis convention: `+x` forward, `+y` left, `+z` up

**⚠️ Critical sign convention:** Positive yaw in the simulator API is **clockwise**. This is opposite to the standard math convention (counter-clockwise positive). The `ArcPlanner` already accounts for this. Do not re-introduce sign flips in new code.

**Pose representation:**
- All poses are 4×4 SE(3) homogeneous transformation matrices (numpy arrays)
- Position: `pose[:3, 3]` — full 3D position
- XY navigation position: `pose[:2, 3]` — only x and y used for planning
- Rotation: `pose[:3, :3]` — rotation matrix
- GTSAM convention: `a_T_b` = transform *from* frame b *into* frame a

**Transform utilities:**
```python
from lac.utils.frames import invert_transform_mat, apply_transform, make_transform_mat
```
- `invert_transform_mat(T)` → 4×4 inverse of an SE(3) matrix using `R^T` block (cheaper than `np.linalg.inv`).
- `apply_transform(T, points)` → applies a 4×4 transform to an `(N, 3)` array of points: `points @ T[:3,:3].T + T[:3,3]`.
- There is **no built-in helper that transforms 2D positions** — for Stretch A you must lift `[x, y]` to `[x, y, 0, 1]` and multiply through `current_pose`, or use `apply_transform(current_pose, arc_xyz)` where `arc_xyz` is an `(N, 3)` array built from the rover-local arc points.

**Map/grid frame:**
- Scene bounds: x ∈ [-20, 20] m, y ∈ [-20, 20] m (40 × 40 m environment) — used for collision/scene logic only.
- Map (and DEM) covers ONLY `[-MAP_EXTENT, MAP_EXTENT] = [-13.5, 13.5]` m, centered on the lander (smaller than the full scene).
- Map grid: 180 × 180 cells, `CELL_WIDTH = 0.15 m` (180 × 0.15 = 27 m = 2 × MAP_EXTENT). ✓
- The geometric map array is shape `(MAP_SIZE, MAP_SIZE, 4)` with channels `[x_coord, y_coord, height_z, rock_bool]`. The first axis indexes **x**, the second indexes **y** — confirmed by `bin_points_to_grid` in `lac/mapping/mapper.py`, which uses `scipy.stats.binned_statistic_2d(x, y, z, …)` (first arg → first axis).
- **Correct world-to-grid conversion (use this — the earlier `SCENE_MIN_*` formula was wrong):**
  ```python
  i = int((x + MAP_EXTENT) / CELL_WIDTH)   # x → first index, range [0, MAP_SIZE)
  j = int((y + MAP_EXTENT) / CELL_WIDTH)   # y → second index, range [0, MAP_SIZE)
  # i.e. map_array[i, j, :] corresponds to world cell (x_center, y_center)
  ```
- Out-of-map cells: any `(x, y)` with `|x| > MAP_EXTENT` or `|y| > MAP_EXTENT` is outside the geometric map. `PerceptionMap` should return 0.0 (or treat as impassable) for these.

---

## 6. Key Parameters from `lac/params.py`

All values below are verified against `lac/params.py`.

```python
LUNAR_GRAVITY = np.array([0.0, 0.0, 1.6220])  # m/s^2  (note: positive z component as written)

FRAME_RATE = 20          # fps — simulation runs at 20 Hz
DT = 1 / FRAME_RATE      # 0.05 s per step
TARGET_SPEED = 0.2       # m/s — rover target speed
ROVER_RADIUS = 0.75      # m — used for collision checking
CELL_WIDTH = 0.15        # m — map grid resolution
MAP_SIZE = 180           # cells per side (180 × 180 grid)
MAP_EXTENT = 13.5        # m — half-extent of the map (covers [-13.5, 13.5] m in x and y)
HEIGHT_ERROR_TOLERANCE = 0.05  # m — geometric map score threshold (also GEOMETRIC_MAP_THRESHOLD in map_utils.py)

SCENE_MIN_X = -20.0      # m   (scene = full simulator extent, larger than the map)
SCENE_MAX_X =  20.0      # m
SCENE_MIN_Y = -20.0      # m
SCENE_MAX_Y =  20.0      # m
SCENE_MAX_Z = 10.0       # m
SCENE_MIN_Z = 0.0        # m

# Lander geometry
LANDER_WIDTH  = 3.0      # m
LANDER_HEIGHT = 3.0      # m
LANDER_GLOBAL            # (4, 3) array — corners of lander bbox in global frame

# Controller / waypoint following
WAYPOINT_REACHED_DIST_THRESHOLD = 1.5   # m — when a waypoint is considered reached
KP_LINEAR = 1.0          # used by NavAgent for v-feedback (also exposed via config.control.kp_linear)
MAX_LINEAR_DELTA = 0.2   # m/s — clip on v adjustment per step
KP_STEER = 0.3
MAX_STEER = 1.2          # rad/s
MAX_STEER_DELTA = 1.0    # rad/s

# Stereo / camera
STEREO_BASELINE = 0.162  # m — stereo camera baseline
IMG_FOV_RAD = 1.22173    # 70 degrees
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FL_X = IMG_WIDTH / (2 * np.tan(IMG_FOV_RAD / 2))
FL_Y = FL_X              # square pixels
CAMERA_INTRINSICS        # 3×3 K matrix built from FL_X, FL_Y, IMG_WIDTH, IMG_HEIGHT

# Rocks / obstacle avoidance (consumed inside arc_planner.py)
ROCK_MIN_RADIUS = 0.08   # m — minimum rock radius considered an obstacle
ROCK_AVOID_DIST = 2.0    # m
ROCK_MASK_AVOID_MIN_AREA = 1000
ROCK_MASK_MAX_AREA = 50000
ROCK_BRIGHTNESS_THRESHOLD = 50
MAX_DEPTH = 56.5         # m

# Arm angles (used in NavAgent.initialize())
ARM_ANGLE_STATIC_RAD = 1.0472          # 60 deg (back arm)
FRONT_ARM_ANGLE_STATIC_RAD = pi/2

# EKF noise parameters (used for covariance propagation)
EKF_SMOOTHING_INTERVAL = 10  # run smoothing every N steps
EKF_INIT_R = 0.001       # initial position std [m]
EKF_INIT_V = 0.01        # initial velocity std [m/s]
EKF_INIT_ANGLE = 0.001   # initial angle std [rad]
EKF_Q_SIGMA_A = 0.03     # process noise: acceleration std
EKF_Q_SIGMA_ANGLE = 0.00005  # process noise: angle std
EKF_R_SIGMAS = np.array([0.25, 0.25, 0.25, 0.05, 0.05, 0.2])  # measurement noise stds
EKF_P0                   # np.diag of squared init stds — 9×9 (pos, vel, angle blocks)

# Tag / fiducial geometry
GEOMETRY_DICT            # loaded from docs/geometry.json — rover camera mounts + lander tags
TAG_LOCATIONS, TAG_GROUP_BEARING_ANGLES, WHEEL_RIG_POINTS
```

Constants you will rely on most heavily in the new planning code: `CELL_WIDTH`, `MAP_SIZE`, `MAP_EXTENT`, `WAYPOINT_REACHED_DIST_THRESHOLD`, `ROVER_RADIUS`, `LANDER_GLOBAL`, `TARGET_SPEED`, `DT`, `FRAME_RATE`, plus the `EKF_*` block.

---

## 7. Existing Planner Architecture — What You Are Replacing

### WaypointPlanner (global layer — YOUR REPLACEMENT)

**File:** `lac/planning/waypoint_planner.py`

**Module-level constants (verified):**
- `WAYPOINT_TIMEOUT = 2000` steps — if a waypoint hasn't been reached in this many steps, the planner force-advances to the next waypoint.
- `SPIRAL_MIN = 3.5`, `SPIRAL_MAX = 5.5`, `SPIRAL_STEP = 1.0` (used only by `trajectory_type="spiral"`).
- `DEFAULT_ORDER = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])` — clockwise quadrant order starting from top-left, also re-exported from `waypoint_generation.py`.

**Interface (verified against the code):**
```python
class WaypointPlanner:
    def __init__(self, initial_pose: np.ndarray,
                 trajectory_type: str = "five_loops",
                 waypoint_reached_threshold: float = WAYPOINT_REACHED_DIST_THRESHOLD)
    # self.waypoints: np.ndarray shape (N, 2) — 2D [x, y] positions in the GLOBAL frame
    # self.waypoint_idx: int — index of the current target
    # self.last_waypoint_step: int — used for WAYPOINT_TIMEOUT logic

    def get_waypoint(self, step: int, pose: np.ndarray, print_progress: bool = False
                    ) -> tuple[np.ndarray | None, bool]:
        # Returns: (waypoint_xy: np.ndarray shape (2,) in global frame, advanced: bool)
        # `advanced=True` is returned when EITHER the current waypoint is reached
        # (Euclidean distance < waypoint_reached_threshold) OR the timeout fires.
        # When all waypoints have been visited, returns (None, True) → NavAgent then
        # calls self.mission_complete().
```

`trajectory_type` valid values (mapped inside `__init__`):
- `"spiral"`     → `gen_spiral(initial_pose, SPIRAL_MIN, SPIRAL_MAX, SPIRAL_STEP)`
- `"five_loops"` → `gen_five_loops(initial_pose, extra_closure=True)`
- `"nine_loops"` → `gen_nine_loops(initial_pose)`
- `"triangles"`  → `gen_triangle_loops(initial_pose, additional_loops=False)`

All four generators return an `(N, 2)` array of waypoints in the **global frame**, where the origin is the lander and `initial_pose` is only consulted to pick the starting quadrant (via `get_starting_direction_order`). Spacing between waypoints is uneven — typically 2–4 m for the loop patterns, 1–3 m for the spiral.

**How it's used in `nav_agent.py`:**
```python
waypoint, advanced = self.planner.get_waypoint(self.step, nav_pose, print_progress=True)
if waypoint is None:
    self.mission_complete()
    return carla.VehicleVelocityControl(0.0, 0.0)
# waypoint is then passed directly to ArcPlanner:
control, path, _ = self.arc_planner.plan_arc(waypoint, nav_pose, data["rock_data"])
```

**Your `PerceptionAwarePlanner` must implement exactly this interface.** The `waypoint` it returns is a `np.ndarray` of shape `(2,)` containing `[x, y]` in the global world frame.

### ArcPlanner (local layer — DO NOT MODIFY except Stretch A)

**File:** `lac/planning/arc_planner.py`

**What it does:** DWA-style local planner. Samples Dubins arcs (pre-computed at init) using angular velocities ω ∈ [-`max_omega`, `max_omega`] rad/s with nonlinear spacing (denser near ω=0, via `nonlinear_linspace(..., power=2)`). For each candidate arc, scores by endpoint distance to the global waypoint. Adds `+1000` penalty to any arc whose points fall inside the lander bbox or within any tracked rock's radius. Returns the lowest-cost arc that is still valid.

**Constructor (verified):**
```python
ArcPlanner(arc_config=41,        # int → 41 single-segment arcs (default used by NavAgent)
                                 # tuple (n1, n2) → "branched" two-segment arcs (n1*n2 arcs)
           arc_duration=8.0,     # s
           max_omega=0.8)        # rad/s
# NUM_ARC_POINTS = int(arc_duration / DT) = int(8.0 / 0.05) = 160 points per arc.
# self.np_candidate_arcs: shape (41, 160, 3) — each row is (x, y, theta) in ROVER-LOCAL frame
#                        because each arc is generated by dubins_traj(np.zeros(3), [v, w], ...).
```

**Key interface (verified):**
```python
control, path, waypoint_local = self.arc_planner.plan_arc(
    waypoint_global: np.ndarray,  # shape (2,) — [x, y] in global frame
    current_pose: np.ndarray,     # 4×4 SE(3) pose
    rock_data: dict               # {'centers': np.ndarray (N, 3) in ROVER-LOCAL frame, z≈0,
                                  #  'radii':   np.ndarray (N,) in meters}
)
# Returns: control = (v, w) Python tuple, where v = TARGET_SPEED (= 0.2 m/s),
#                                              w is the ω of the chosen arc, BUT then re-scaled
#                                              by 1/self.scale = 2 when populated into self.root_vw
#                                              (see line `self.root_vw.append((v, w * 1 / self.scale))`).
#          path = self.np_candidate_arcs[i_best] — shape (160, 3) in rover-local frame.
#          waypoint_local = pose_inv @ [wx, wy, 0, 1] — the waypoint expressed in the rover frame.
# Returns (None, None, None) if every arc collides — NavAgent treats this as "stuck" and runs the
# backup maneuver.
```

**Coordinate frame at the scoring step (CRITICAL for Stretch A):**
- Candidate arcs are stored in **rover-local frame** in `self.np_candidate_arcs`. They are NOT transformed to world frame before scoring.
- The current scoring line is:
  ```python
  path_costs = np.linalg.norm(self.np_candidate_arcs[:, -1, :2] - waypoint_local[:2], axis=1)
  ```
  i.e. it compares endpoints to `waypoint_local` (the waypoint expressed in rover frame).
- To add a perception term in `plan_arc()`, you must transform each arc's sampled XY points from rover-local to world frame before querying `PerceptionMap`. The cheapest way is:
  ```python
  # arc_xy_local: (P, 2)  →  lift to (P, 3) with z=0
  arc_xyz_local = np.concatenate([arc[:, :2], np.zeros((arc.shape[0], 1))], axis=1)
  arc_xyz_world = apply_transform(current_pose, arc_xyz_local)   # uses existing util
  # Then sample perception_map at arc_xyz_world[:, :2]
  ```

**What this means for your planner:** Your planner generates *global waypoints* spaced ~1.5–3 m apart. The ArcPlanner handles all fine steering, rock avoidance, and velocity control. You never output velocity commands directly.

### 7.1 SLAM Frontend and Backend — Interfaces Used by the Planner

These are the inherited modules. Read carefully — Stretch B and C rely on these outputs.

**`Frontend.process_frame(images_dict) → data`** (`lac/slam/frontend.py`):
The `images_dict` is mutated in place and returned as `data`. After the call it contains (at minimum):

| Key | Type / shape | Notes |
|---|---|---|
| `FrontLeft`, `FrontRight` | `np.ndarray (H, W)` or `(H, W, 3)` | input images |
| `BackLeft`, `BackRight` | same | only when `BACK_CAMERAS=True` |
| `step` | int | passed in by NavAgent |
| `imu_measurements` | deque of IMU samples since last image | passed in by NavAgent |
| `prev_pose` | `np.ndarray (4, 4)` | passed in by NavAgent |
| `odometry` | `np.ndarray (4, 4)` | relative SE(3) odometry produced by VO (or IMU fallback) |
| `odometry_source` | `"VO"` or `"IMU"` | which path was used |
| `keyframe` | `bool` | `data["step"] % KEYFRAME_INTERVAL == 0`, KEYFRAME_INTERVAL = 20 |
| `tracked_points` | `TrackedPoints` | front-cam feature tracks (used by Backend) |
| `back_tracked_points` | `TrackedPoints` | optional back-cam tracks |
| `rock_data` | `{"centers": np.ndarray (N, 3) in rover-local frame, "radii": np.ndarray (N,)}` | the dict the ArcPlanner consumes |
| `rock_depth` | list of dicts (per-rock stereo depth results) | used by mapper for rock projection |
| `left_pred` | `np.ndarray (H, W)` int — segmentation prediction | used for rock overlay in rerun viz |

**`Backend.update(data)`** consumes the dict above. Loop closure detection is triggered automatically whenever `data["keyframe"]` is True and at least `LOOP_CLOSURE_EXCLUDE = 10` keyframes exist. There is **no public method to force a loop closure check** outside this keyframe path.

**`Backend.get_trajectory() → list[np.ndarray (4,4)]`**
- Returns the full optimized trajectory as a Python list of 4×4 SE(3) matrices.
- Includes the initial pose at index 0; one pose per `backend.update(data)` call afterwards.
- Backend is only updated on steps where image data is available and `step >= ARM_RAISE_WAIT_FRAMES`, so the list length is roughly `(step - 80) / 2`.

**`Backend.project_point_map() → SemanticPointCloud`** (the Stretch B data source):
```python
@dataclass
class SemanticPointCloud:
    points: np.ndarray  # shape (N, 3) — XYZ in the WORLD frame
    labels: np.ndarray  # shape (N,)   — int per SemanticClasses enum (GROUND, ROCK, …)
```
Internally, each anchor keyframe's local points are reprojected with the latest optimized pose, then stacked. NavAgent already calls this inside `update_map()`.

**`Backend.get_state() → dict`** (the Stretch C validation source):
```python
{
    "odometry":          np.ndarray (M, 4, 4),  # per-step relative odometry
    "odometry_sources":  np.ndarray (M,) int,   # 0 = VO, 1 = IMU
    "loop_closures":     np.ndarray (K, 2) int, # pairs (anchor_pose_idx, current_pose_idx)
    "loop_closures_poses": np.ndarray (K, 4, 4),  # relative SE(3) of each loop closure
                                                # NOTE the key is the plural "loop_closures_poses"
                                                # (NOT "loop_closure_poses"). NavAgent.finalize()
                                                # already remaps it to "loop_closure_poses" before
                                                # saving — when reading from a live agent, use the
                                                # plural form returned by get_state().
}
```
The backend also stores `self.loop_closures_matches` (number of feature matches per LC) and `self.keyframe_traj` (np.ndarray of past keyframe poses), but these are not in `get_state()`. Past *keyframe poses* are the most useful signal for the LoopClosureMap.

**Loop closure constraints (Stretch C — important for LC map design):**
- Loop closures are only attempted at keyframes (every `KEYFRAME_INTERVAL = 20` steps, i.e. every ~1 s).
- The most recent `LOOP_CLOSURE_EXCLUDE = 10` keyframes are skipped → minimum lookback of ~200 steps.
- Candidate keyframes must be within `LC_DIST_THRESHOLD` (config `loop_closure.distance_threshold_m`, default 1.0 m, 0.75 m for nine_loops) in XY and within `LC_ANGLE_THRESHOLD` (5°) in attitude.
- After geometric filtering, an image-matching call (`keyframe_estimate_loop_closure_pose`) must produce ≥ `min_matches` (default 100) matches with score ≥ `min_score` (default 0.5). Otherwise the closure is dropped.
- The LoopClosureMap should therefore only score cells within ~`MAP_EXTENT` that (a) have a past keyframe pose nearby (≲ 1 m), (b) lie ≳ 200 steps in the past, and (c) ideally have non-trivial ρ.

**Localization uncertainty / covariance — NOT exposed:** Neither `get_state()` nor `get_trajectory()` returns marginal covariances from GTSAM. There is no helper that extracts the marginal Σ from the optimized factor graph. Implications:
- The "actual" localization uncertainty cannot be read directly from the backend.
- The `EKFCovariancePropagator` we build (Section 8.2) must run as an independent shadow filter for planning purposes, not be initialized from the backend's posterior.
- For Stretch C validation we compare RMSE vs. ground truth before/after each loop closure (already supported by `positions_rmse_from_poses` in `lac/util.py` + `slam_eval_poses`).

### 7.2 NavAgent.run_step Data Flow (verified end-to-end)

Per step `run_step(input_data)` executes the following (omitting logging):

1. `self.step += 1`; abort if `step > MISSION_TIMEOUT = 30000`.
2. Read `ground_truth_pose` (unless `EVAL`) — used as `nav_pose` only if `USE_GROUND_TRUTH_NAV=True`. By default `nav_pose = self.current_pose`, which is the SLAM-estimated pose from the **previous** step's backend output.
3. `waypoint, advanced = self.planner.get_waypoint(self.step, nav_pose, …)`. If `waypoint is None` → `self.mission_complete()` and return zero control.
4. **If `advanced` is True** (waypoint reached or timed out) → `agent_map = self.update_map()` (scores and logs).
5. Append IMU sample.
6. **If `image_available()` (i.e. `step % 2 == 0`) AND `step >= ARM_RAISE_WAIT_FRAMES (= 80)`:**
   - On step 80: `self.frontend.initialize(images)` (no backend update, no control).
   - Otherwise: `data = self.frontend.process_frame(images)`; `self.backend.update(data)`.
   - Then `control, path, _ = self.arc_planner.plan_arc(waypoint, nav_pose, data["rock_data"])`.
   - If `control is not None`: unpack `(v, w)`, apply proportional feedback `v += kp_linear * (TARGET_SPEED - ||current_velocity||)` clipped to `±MAX_LINEAR_DELTA`.
7. On **odd steps** (or before frame 80, or on the init frame): `control` remains the default `(0.0, 0.0)` from this iteration BUT `self.current_v / self.current_w` retain their previous values, so the final `carla.VehicleVelocityControl(self.current_v, self.current_w)` keeps driving with the last-computed command. Image-less steps do not stop the rover.
8. Carla control selection:
   - `step < ARM_RAISE_WAIT_FRAMES` → stop (zero velocity command).
   - Backup maneuver active or `check_stuck()` triggers → `run_backup_maneuver()`.
   - `control is None` (no safe arc found this step) → `run_backup_maneuver()`.
   - Otherwise → `carla.VehicleVelocityControl(self.current_v, self.current_w)`.
9. Update state: `slam_poses = self.backend.get_trajectory(); self.current_pose = slam_poses[-1]; self.current_velocity = self.frontend.current_velocity`.

**Stuck detection (`check_stuck`)**: enabled after step `ARM_RAISE_WAIT_FRAMES + 10`. Fires if `||current_velocity|| < 0.25 * TARGET_SPEED` for >50% of a 2–3 s window.

**`update_map()` — verified contents:**
```python
def update_map(self):
    g_map = self.get_geometric_map()             # inherited from AutonomousAgent
    map_array = g_map.get_map_array()            # np.ndarray (180, 180, 4): [x, y, z, rock_bool]
    semantic_points = self.backend.project_point_map()  # SemanticPointCloud in WORLD frame
    map_array = process_map(semantic_points, map_array,
                            rock_count_thresh=self.config["mapping"]["rock_count_thresh"])
    return map_array.copy()
```
- Called only when `advanced=True` (a waypoint is reached or times out) and once from `finalize()`. **Not** called every step.
- At call time, `self.current_pose` is the SLAM pose from the END of the previous `run_step`. `nav_pose` was computed from the same value, so the planner and `update_map` see the same pose. The backend trajectory does NOT yet include this step's pose (that update happens later in the step, after the arc planner runs).
- Available data inside `update_map()` for Stretch B / C: `self.backend.project_point_map()` (semantic point cloud in world frame), `self.backend.get_trajectory()` (full pose history through previous step), `self.backend.get_state()` (loop closures + odometry sources), `self.current_pose`, `self.step`, `self.config`. Ground-truth pose is also captured via `self.slam_eval_poses` (only when `LOG_DATA=True`).

---

## 8. Technical Design: What You Are Building

### 8.1 PerceptionMap (`lac/planning/perception_map.py`)

**Purpose:** Given a DEM heightmap and sun angle, compute a predicted visual feature density ρ(x,y) ∈ [0,1] for every cell in the map grid.

**Inputs:**
- `heightmap`: `np.ndarray` of shape `(H, W)` — elevation values in meters, aligned with map grid
- `cell_width`: float (= `CELL_WIDTH = 0.15 m`)
- `sun_azimuth_deg`: float — sun direction in the horizontal plane (degrees from +x axis)
- `sun_elevation_deg`: float — sun elevation above horizon (degrees)

**Algorithm:**
1. **Roughness:** For each cell, compute std of elevation in a local window (5×5 cells recommended). High roughness → high expected feature count. Normalize to [0,1].
2. **Shadow mask:** Ray-cast from each cell toward the sun. A cell is shadowed if any intervening terrain is taller (accounting for distance and sun elevation). Shadowed cells → ρ = 0.
3. **Sun modulation:** Scale roughness by a sun-elevation factor. At very low sun elevation (south pole conditions, ~1–5°), long shadows *enhance* feature detection of microrelief. Proposed model: `sun_factor = np.clip(np.sin(sun_elev_rad) * 10, 0.1, 1.0)` (peaks at ~6° elevation and above, still non-zero at grazing angles).
4. **Combine:** `rho = roughness_normalized * sun_factor * (1 - shadow_mask)`. Normalize to [0,1].

**Interface:**
```python
class PerceptionMap:
    def __init__(self, heightmap, cell_width, sun_azimuth_deg, sun_elevation_deg,
                 roughness_window=5)
    
    def get_density_grid(self) -> np.ndarray:
        # Returns (H, W) array of ρ values in [0,1] aligned with map grid
    
    def get_feature_density(self, x: float, y: float) -> float:
        # Query ρ at a world-frame (x, y) position
        # Returns 0.0 if out of bounds
    
    def get_uncertainty_cost(self, x: float, y: float) -> float:
        # Returns 1 - ρ(x, y) — high uncertainty where features are sparse
    
    def visualize(self, save_path: str = None):
        # Plot roughness, shadow mask, and final ρ map side by side
```

**Stub / fallback:** If no DEM is available yet, initialize with a flat heightmap + no shadows. All cells get uniform ρ = 0.5. The planner degrades gracefully to approximately a shortest-path planner.

**Validation (key experimental result for paper):**  
Run ORB or SuperPoint feature detector on rendered LAC frames. For each frame, record: (a) rover position, (b) feature count. Compute Pearson correlation between DEM-predicted ρ at rover position and observed feature count. Report scatter plot + r value. This is Section 5 (Results) material regardless of planner performance.

### 8.2 EKF Covariance Propagation (`lac/planning/ekf_covariance.py`)

**Purpose:** Propagate predicted localization uncertainty along a candidate path, given the feature density model. Used by the planner to score paths.

**State:** The EKF state is 9D: `[x, y, z, vx, vy, vz, roll, pitch, yaw]`. For planning purposes, we care about the 2D positional uncertainty (x,y block of the covariance).

**Prediction step** (dead-reckoning between observations):
```
P_{k+1|k} = F_k P_k F_k^T + Q_k
```
where `F_k` is the motion Jacobian (approximately identity for slow rover motion over short steps) and `Q_k` is the process noise matrix built from `EKF_Q_SIGMA_A` and `EKF_Q_SIGMA_ANGLE`.

**Update step** (when observations are available):
```
Λ_{k+1} = Λ_{k+1|k} + g(ρ_k) · I_nominal
```
where `Λ = P^{-1}` is the information matrix, `g(ρ_k)` scales the nominal information increment by predicted feature density at cell k, and `I_nominal` is the nominal information block derived from `EKF_R_SIGMAS`.

**D-optimality score** (the objective — higher is better localization):
```
score = log(det(Λ_N))
```
where `Λ_N` is the information matrix at the end of the path. D-optimality is used (not A- or E-optimality) because Carrillo et al. (2018) proved it is the only standard criterion that remains monotonically increasing as uncertainty grows during dead-reckoning.

**Interface:**
```python
class EKFCovariancePropagator:
    def __init__(self, P0: np.ndarray = None)  # defaults to params.EKF_P0
    
    def propagate_path(self, path_xy: np.ndarray, density_fn: callable,
                       step_size: float = 0.15) -> float:
        # path_xy: (N, 2) array of world-frame waypoints
        # density_fn: callable(x, y) → ρ ∈ [0,1]
        # Returns: D-optimality score = log(det(Λ_N)) — higher = better
    
    def get_final_covariance(self, path_xy: np.ndarray, density_fn: callable) -> np.ndarray:
        # Returns final 9×9 covariance matrix P_N
```

### 8.3 PerceptionAwarePlanner (`lac/planning/perception_aware_planner.py`)

**Purpose:** Drop-in replacement for `WaypointPlanner` that produces waypoints maximizing the combined path cost + localization quality objective.

**Objective — full three-term formulation:**
```
J(ξ) = α·C_path(ξ) + β·U(ξ) - γ·L(ξ)
```
where:
- `C_path(ξ)` = total Euclidean path length
- `U(ξ)` = predicted localization uncertainty = negative D-optimality score (high where ρ is low)
- `L(ξ)` = loop closure opportunity score (high where path passes near previously-mapped, feature-rich locations) — zero until Stretch C is implemented
- α, β, γ are tunable weights; framed as Lagrange multipliers in the paper's Problem Statement section

**Per-edge A* cost** when moving from cell u to adjacent cell v:
```python
edge_cost(u, v) = alpha * dist(u, v)                      # path length
                + beta  * perception_map.get_uncertainty_cost(v)  # 1 - rho(v)
                - gamma * loop_closure_map.get_lc_score(v)        # LC opportunity
```

This makes the planner avoid cells with low feature density (high uncertainty cost), while still preferring shorter paths when perception quality is comparable.

**Goal handling:** For the LAC challenge, the rover has no explicit single goal — it must maximize map coverage. The planner should generate a *coverage trajectory* that visits high-ρ regions of the map, analogous to how the pre-baked `gen_five_loops` generates a coverage pattern. The key difference is that your planner's coverage is perception-weighted: it spends more traversal time in high-ρ terrain.

**Recommended approach:** Generate a sequence of sub-goals using a greedy or TSP-lite policy over the high-ρ cells of the map, then plan A* segments between consecutive sub-goals.

**Interface — must match WaypointPlanner exactly:**
```python
class PerceptionAwarePlanner:
    def __init__(
        self,
        initial_pose: np.ndarray,           # 4×4 SE(3)
        perception_map: PerceptionMap,
        alpha: float = 1.0,                 # path cost weight
        beta: float = 2.0,                  # uncertainty cost weight
        gamma: float = 0.0,                 # loop closure incentive weight (0 = disabled)
        loop_closure_map = None,            # LoopClosureMap instance (Stretch C); None = disabled
        waypoint_spacing: float = 2.0,      # m between waypoints along planned path
        waypoint_reached_threshold: float = WAYPOINT_REACHED_DIST_THRESHOLD,
    )
    
    def get_waypoint(self, step: int, pose: np.ndarray,
                     print_progress: bool = False) -> tuple[np.ndarray | None, bool]:
        # Identical interface to WaypointPlanner.get_waypoint()
        # Returns (waypoint_xy: shape (2,), advanced: bool)
        # Returns (None, True) when coverage is complete

    def replan(self, current_pose: np.ndarray, updated_perception_map: PerceptionMap,
               updated_lc_map=None) -> None:
        # Triggered by Stretch B (online refinement) and Stretch C (LC update)
        # Replans A* from current position; preserves already-visited waypoints
```

**Internal state:**
```python
self.waypoints      # list of (2,) arrays — planned global waypoints
self.waypoint_idx   # current target index
self.last_waypoint_step  # for timeout logic (copy from WaypointPlanner)
```

### 8.4 PerceptionAwareAgent (`agents/perception_aware_agent.py`)

A fork of `nav_agent.py` with minimal changes in `setup()` and `update_map()`:

```python
# setup() changes:
from lac.planning.perception_aware_planner import PerceptionAwarePlanner
from lac.planning.perception_map import PerceptionMap
from lac.planning.loop_closure_map import LoopClosureMap  # Stretch C only

heightmap = np.load("data/lac_heightmap.npy")
self.perception_map = PerceptionMap(
    heightmap=heightmap,
    cell_width=params.CELL_WIDTH,
    sun_azimuth_deg=self.config["sun"]["azimuth_deg"],
    sun_elevation_deg=self.config["sun"]["elevation_deg"],
)
self.planner = PerceptionAwarePlanner(
    initial_pose=self.initial_pose,
    perception_map=self.perception_map,
    alpha=self.config["planning"]["alpha"],
    beta=self.config["planning"]["beta"],
    gamma=self.config["planning"].get("gamma", 0.0),  # 0 until Stretch C
)

# update_map() addition (Stretch B + C):
# After existing map update logic, add:
self.perception_map.update_from_slam(self.backend.project_point_map(), self.current_pose)
lc_map = LoopClosureMap.from_trajectory(
    self.backend.get_trajectory(),  # list of 4x4 SE(3) — all past optimized poses (incl. initial)
    self.perception_map,
)
self.planner.replan(self.current_pose, self.perception_map, lc_map)
```

Notes:
- `update_map()` is only invoked when a waypoint is reached (or times out), not every step — so Stretch B/C replanning fires at roughly 1.5–3 m of rover travel intervals, not at SLAM rate.
- `self.current_pose` at this point reflects the previous step's optimized SLAM pose; the trajectory list from `get_trajectory()` ends at that same pose. The current step's odometry has not yet been integrated.

Everything else — Frontend, Backend, ArcPlanner, control loop, backup maneuver, stuck detection, Rerun logging, `finalize()` — is **completely unchanged**.

---

### 8.5 Stretch A — Perception-Aware Local Arc Selection

**File:** Minimal augmentation to `lac/planning/arc_planner.py`

**What it does:** Adds a small perception term to the arc scoring in `plan_arc()` so the local planner also prefers passing through feature-rich cells, when multiple arcs are otherwise comparable in distance to the waypoint.

**Modified cost in `plan_arc()`:**
```python
# Current cost (distance to waypoint endpoint only):
path_costs = np.linalg.norm(self.np_candidate_arcs[:, -1, :2] - waypoint_local[:2], axis=1)

# Augmented cost (add perception penalty along arc):
# self.np_candidate_arcs is shape (N_arcs, N_points, 3) in ROVER-LOCAL frame.
# Lift to (N_points, 3) by setting z=0 and transform with current_pose.
for i, arc in enumerate(self.np_candidate_arcs):
    arc_xyz_local = np.concatenate([arc[:, :2], np.zeros((arc.shape[0], 1))], axis=1)
    arc_xyz_world = apply_transform(current_pose, arc_xyz_local)  # (N_points, 3)
    mean_uncertainty = np.mean([perception_map.get_uncertainty_cost(p[0], p[1])
                                for p in arc_xyz_world[::5]])     # sample every 5 points
    path_costs[i] += arc_perception_weight * mean_uncertainty
```

The transform step is mandatory because at the scoring point in `plan_arc()` the candidate arcs are still in the rover-local frame — see Section 7 (ArcPlanner) for the verified frame.

**Key design decisions:**
- The perception term is *additive and soft* — it never overrides collision avoidance (rocks add +1000, perception adds O(0.1–1.0))
- `arc_perception_weight` is a new config parameter (suggest starting at 0.5)
- Guarded by `use_perception_arcs: bool` flag in config for clean ablation
- `perception_map` reference passed into `ArcPlanner` constructor; `None` = original behavior

**Paper contribution:** Enables the ablation "global perception planning + local perception arc selection vs. global only," showing perception-awareness at two scales.

---

### 8.6 Stretch B — Online ρ Refinement

**Trigger:** Every call to `update_map()` in `nav_agent.py` — fires each time a waypoint is reached (~every 1.5–3 m of travel).

**Algorithm:**
1. Pull SLAM semantic point map: `semantic_points = backend.project_point_map()`
2. For each point in the map, count observed features per grid cell → `rho_observed(x,y)`
3. Blend with DEM prior: `rho_blended = (1 - w) * rho_prior + w * rho_observed`, where `w` increases from 0→1 as more observations accumulate (e.g., `w = min(n_obs / n_obs_saturation, 1.0)`)
4. Call `planner.replan(current_pose, updated_perception_map)` to recompute A* from current position

**New method on `PerceptionMap`:**
```python
def update_from_slam(self, semantic_points, current_pose, blend_weight: float = None):
    # Updates self._rho_grid by blending prior with observed feature counts
    # blend_weight=None → use adaptive weight based on observation count
```

**Paper contribution:** Closes the prior→observation→replanning loop. Gives a third experimental condition (prior-only vs. observed-only vs. blended) and validates that the DEM-derived prior is a useful initialization even when observations are available.

---

### 8.7 Stretch C — Loop Closure Incentivization

**Motivation:** Perception-aware routing (Stretch A/B) reduces *drift accumulation* by keeping the rover in feature-rich terrain. Loop closure incentivization *corrects drift* by proactively routing the rover back through previously mapped locations, triggering the SLAM backend's loop closure detection.

**Objective term:**
```
-γ · L(ξ)   where L(ξ) = loop closure opportunity score along path ξ
```

**Loop Closure Opportunity Map — `lac/planning/loop_closure_map.py`:**

Precomputed at each replanning event as a 180×180 grid:
```python
LC_map(x, y) = rho(x, y)                  # high features → recognizable
             * was_visited(x, y)           # must have been seen before
             * recency_penalty(x, y)       # penalize recently visited cells
             * (1 - proximity_penalty)     # avoid immediate re-visitation
```

Where:
- `was_visited`: binary mask from past trajectory history (`backend.get_trajectory()`)
- `recency_penalty`: sigmoid decay based on steps since last visit — cells visited >200 steps ago score full credit
- `proximity_penalty`: suppresses cells within 3 m of current position (too close to be a meaningful loop closure)

**Interface:**
```python
class LoopClosureMap:
    @classmethod
    def from_trajectory(cls, past_poses: list, perception_map: PerceptionMap) -> 'LoopClosureMap'
    
    def get_lc_score(self, x: float, y: float) -> float
        # Returns LC opportunity score ∈ [0, 1] at world position (x, y)
    
    def get_lc_grid(self) -> np.ndarray
        # Returns (180, 180) array for visualization and A* lookup
```

**Integration into A* edge cost:**
```python
edge_cost(u, v) = alpha * dist(u, v)
                + beta  * perception_map.get_uncertainty_cost(v)
                - gamma * lc_map.get_lc_score(v)
```

**Implementation choice — A* greedy vs. MPC (read before implementing):**

Two formulations are viable for the loop closure incentive. The chosen approach is **A* greedy** (the γ term above). An MPC-style two-point boundary value formulation was also considered and explicitly rejected. The trade-off is documented here for the paper's design decisions section:

> **A* greedy (chosen):**  
> The LC opportunity score is added as a negative cost term on each grid cell during the existing A* search. No new solver or optimization loop is needed — the same planner that handles perception-awareness also handles LC incentivization. The LC map is precomputed as a 180×180 grid lookup, adding negligible per-edge cost during search. The incentive is *soft*: if no high-scoring LC opportunity is near the nominal path, γ·L(v) ≈ 0 and the planner behaves as before. γ = 0 is a clean ablation condition.  
> *Trade-off:* The greedy A* cannot explicitly plan a two-leg detour (divert → LC site → resume coverage). It routes toward LC opportunities only insofar as they align with the overall coverage objective. For most cases in the 40×40 m LAC environment, this is sufficient since past trajectory cells are always nearby.

> **MPC two-point boundary value (not implemented — documented for paper):**  
> An MPC formulation would explicitly plan a short-horizon trajectory from the current position to a target LC site and back to the nominal coverage path, with the detour cost penalized against the expected drift correction benefit. This is more principled for cases where the optimal LC site requires a deliberate detour that A* would not naturally produce. However, it requires: (1) a differentiable cost landscape (the LC map has sharp edges), (2) a separate optimization solve at each replanning event (adding latency), and (3) integration of a second solver alongside A*. Given that the ArcPlanner already handles local dynamic feasibility and the LAC environment is small enough that LC sites are rarely more than 2–3 m off the nominal path, the added complexity is not justified. MPC-style LC planning is noted as a direction for future work in the paper.

**Validation:** Count actual loop closure events from `backend.get_state()['loop_closures']` (shape `(K, 2)`) across missions with γ=0 (no incentive) vs. γ>0 (incentivized). Report loop closure count, timing, and RMSE before/after each closure event. A positive correlation between γ>0 and loop closure frequency is a direct validation of the incentive mechanism.

**Backend constraints discovered (incorporate into LC map construction):**
- The backend only attempts loop closures at keyframes (`KEYFRAME_INTERVAL = 20` steps). This bounds how often γ matters.
- The minimum lookback is `LOOP_CLOSURE_EXCLUDE = 10` keyframes ≈ 200 steps. Mark `was_visited(x, y)` only for keyframes whose pose index is ≥ 10 keyframes in the past.
- A candidate keyframe must lie within `LC_DIST_THRESHOLD = 1.0 m` (or 0.75 m for the nine_loops config) of the rover at trigger time and within 5° in attitude. The LC map should therefore concentrate score in narrow bands around past keyframe poses (≈ 1 m radius), not broad regions.
- Even after geometric filtering, the LC backend can reject a closure if the LightGlue match (`min_score = 0.5`, `min_matches = 100` in the default config) fails. This is uncorrelated with our planner — the γ term is an *opportunity* score, not a guaranteed closure.
- There is no API to force a LC check. The planner cannot trigger closures directly; it can only route the rover through cells where the backend will naturally attempt them.

---

---

## 9. Experimental Design

### 9.1 Baseline Comparisons

> **Superseded by Section 13.A** — see status note above. The original coverage baselines below are kept only for historical reference. The actual baseline is now the slope-aware A* from AA278 HW3 P4 (`data/Example_Implementations/HW3_Final/AA278_2026_HW3_P4.ipynb` + `supplemental/dem.py` + `supplemental/util.py`). The actual comparison is **goal-to-goal SLAM RMSE: P4 slope-aware A* vs. perception-aware A* (ours)**.

Historical table (no longer the experimental design):

| Planner | Description |
|---|---|
| **Baseline 1** | `WaypointPlanner` with `trajectory_type="five_loops"` — original Stanford NavLab coverage pattern |
| **Baseline 2** | `WaypointPlanner` with `trajectory_type="spiral"` — second pre-baked pattern |
| **Ours (core)** | `PerceptionAwarePlanner`, prior ρ only, γ=0, no arc perception (α=1.0, β=2.0) |
| **Ours + Stretch A** | Core + perception-weighted arc selection |
| **Ours + Stretch B** | Core + online ρ refinement via `update_map()` |
| **Ours + Stretch C** | Core + B + loop closure incentivization (γ>0) |
| **Full system** | All stretches combined |

### 9.2 Metrics

**Primary (localization quality):**
- Predicted D-optimality score `log(det(Λ_N))` along planned path — computed offline before rollout
- Actual SLAM localization RMSE vs. ground truth — from `positions_rmse_from_poses()` in `nav_agent.finalize()`
- EKF covariance trace `tr(P_N)` at end of mission
- Loop closure event count — from `backend.get_state()['loop_closures']` (Stretch C metric)
- RMSE before/after each loop closure event (Stretch C metric)

**Secondary (efficiency):**
- Total path length (m)
- Path length overhead vs. shortest-path baseline (% longer)
- Mission completion time (steps)

**Validation (standalone subresult):**
- Pearson r between DEM-predicted ρ(x,y) and ORB/SuperPoint feature count at each frame
- Scatter plot of predicted vs. observed feature density

### 9.3 Ablation Studies

1. **β=0, γ=0** (α only): Pure shortest-path A* — isolates perception weighting contribution
2. **β=∞, γ=0** (perception only): Greedily maximize ρ regardless of path length
3. **Roughness only** (no shadow mask): Shows contribution of illumination modeling
4. **Shadow only** (no roughness): Shows contribution of terrain geometry
5. **α/β sweep**: Plot RMSE vs. path length overhead as β/α increases
6. **γ=0 vs. γ>0** (Stretch C): Loop closure count and RMSE with and without LC incentive
7. **Prior-only vs. blended ρ** (Stretch B): Validates that DEM prior is a useful initialization
8. **Arc perception on/off** (Stretch A): Global-only vs. global + local perception weighting

---

## 10. Literature — Key References

All of these must appear in the paper's Related Work section.

| Reference | Why it matters |
|---|---|
| Placed et al. 2023, "A Survey on Active SLAM," IEEE T-RO | Canonical active SLAM survey; establishes the field |
| Carrillo et al. 2018 | Proves D-optimality monotonicity — justifies our choice of information criterion |
| Cadena et al. 2016, "Past, Present, and Future of SLAM," IEEE T-RO | Broader SLAM context |
| Kim & Eustice 2013/2015, Perception-Driven Navigation | Closest terrestrial analog to our approach; saliency-based information gain |
| Lehner et al. (DLR) 2019, 2023 | Closest planetary analog; perception-aware planning for Mars rovers; no illumination model |
| Dai et al. 2026, Stanford NAVLAB | TA Adam's paper; DEM-anchored lunar stereo SLAM — we build on this as reactive baseline |
| ShadowNav, JPL 2024 | Crater-rim landmarks under active illumination — reactive lunar localization |

**Gap statement (use verbatim in paper):**  
*"Existing lunar SLAM work makes the rover robust to whatever images it encounters; this work plans paths so the rover encounters better images to begin with."*

---

## 11. Paper Structure (ION Format, 6–8 pages solo)

```
1. Introduction
   - Motivation: Artemis / 100+ planned missions, no GPS, comm latency
   - Problem: Vision SLAM degrades in lunar conditions (shadows, low texture)
   - Contributions (three-level system):
       (1) Proactive perception-aware global planning — reduces drift accumulation
       (2) Online ρ refinement — closes the prior→observation→replanning loop
       (3) Loop closure incentivization — corrects drift when it occurs
   - Overview of approach and system architecture

2. Related Work
   - Active SLAM (Placed 2023, Carrillo 2018)
   - Perception-aware planning (Kim & Eustice 2015, Lehner 2019/2023)
   - Lunar surface navigation (Dai 2026, ShadowNav 2024)
   - Gap: none combines prior DEM + illumination + proactive planning + LC incentivization

3. Problem Statement
   - Full objective: J(ξ) = α·C_path + β·U(ξ) - γ·L(ξ)
   - α, β, γ framed as Lagrange multipliers (not ad hoc weights)
   - EKF covariance propagation model
   - D-optimality criterion and why (Carrillo monotonicity proof)
   - Loop closure opportunity score definition
   - Assumptions: known coarse DEM, quasi-static sun, single rover

4. Approach
   - System architecture diagram (4 layers: SLAM backend inherited,
     global planner ours, local ArcPlanner augmented, online refinement)
   - PerceptionMap: roughness + shadow casting + sun modulation + online blend
   - Perception-aware A*: three-term edge cost formulation
   - EKF covariance propagation along path
   - Stretch A: perception-weighted arc scoring
   - Stretch B: online ρ update via update_map() hook
   - Stretch C: loop closure map construction + γ term; design choice
     justification (A* greedy chosen over MPC — see Section 4.x)
   - Integration with LAC / nav_agent

5. Results
   5.1 Feature density predictor validation (Pearson r, scatter plot)
   5.2 Planned path comparison (maps showing baseline vs. each variant)
   5.3 Predicted uncertainty comparison (D-optimality scores)
   5.4 Actual SLAM RMSE comparison across all conditions
   5.5 Loop closure analysis: event count and RMSE correction (γ=0 vs. γ>0)
   5.6 Path length overhead
   5.7 Ablation study (all 8 conditions from Section 9.3)

6. Conclusion
   - Three-level system: reduce accumulation, refine model, correct drift
7. Future Work
   - MPC-style explicit LC detour planning
   - Learned ρ predictor (neural field)
   - Multi-mission sun-angle adaptation
```

---

## 11.5 Config Fields and New Additions

The default config files (`configs/{config, five_loops, nine_loops, spiral, triangles}.json`) currently expose only:
```json
{
  "planning":     { "trajectory": "...", "waypoint_reached_threshold_m": 1.5 },
  "loop_closure": { "min_score": 0.5, "min_matches": 100,
                    "distance_threshold_m": 1.0, "angle_threshold_deg": 5.0 },
  "control":      { "kp_linear": 1.0 },
  "mapping":      { "rock_count_thresh": 25 }    // only present in nine_loops.json
}
```

The new agent (`agents/perception_aware_agent.py`) needs the following additional fields. Add a new config file `configs/perception_aware.json` (or extend `config.json`) containing:

```json
{
  "planning": {
    "trajectory": "perception_aware",            // new value handled by PerceptionAwarePlanner
    "waypoint_reached_threshold_m": 1.5,
    "alpha": 1.0,                                // path cost weight
    "beta":  2.0,                                // uncertainty cost weight
    "gamma": 0.0,                                // loop closure incentive weight (Stretch C)
    "waypoint_spacing_m": 2.0
  },
  "perception_map": {
    "roughness_window": 5,
    "heightmap_path": "data/lac_heightmap.npy"
  },
  "sun": {
    "azimuth_deg":  45.0,
    "elevation_deg": 3.0
  },
  "arc_planner": {
    "use_perception_arcs": false,                // Stretch A flag
    "arc_perception_weight": 0.5                 // Stretch A weight
  },
  "loop_closure": { ... },                       // unchanged from baseline
  "control":      { "kp_linear": 1.0 },
  "mapping":      { "rock_count_thresh": 25 }
}
```

The existing `loop_closure` block is consumed unchanged by `Backend.__init__` and does NOT need new fields for Stretch C — the LoopClosureMap weight γ is part of `planning.*`. The `rock_count_thresh` field must be present (NavAgent reads `self.config["mapping"]["rock_count_thresh"]`); it is missing from most stock configs and should be added when forking.

---

## 12. Environment Setup

**Authoritative procedure: see `docs/SIM_STARTUP.md`.** It is now the source of truth for the install, including all WSL-specific gotchas discovered during the 22.04 → 24.04 migration. The summary below is provided for quick orientation only — do not follow it as a recipe; use SIM_STARTUP.md.

**Verified working configuration (as of 2026-05-26, `vader.stanford.edu` — PRIMARY):**
- Miniconda env `lac` with Python 3.10.20, `PYTHONNOUSERSITE=1` permanently set.
- PyTorch 2.4.1+cu121 (CUDA 12.1), `torch.cuda.is_available()` confirmed True on the RTX 4090.
- All deps from this repo's `requirements.txt` plus `imageio`, `munch`, `segmentation-models-pytorch`, `opt_einsum` (four undocumented deps), plus `astropy==5.2.2` and `lunarsky==0.2.1` (also missing from requirements.txt; needed by `mission_weather.py`).
- `cmake<4` pinned; `apriltag` installed with `--no-build-isolation`.
- LightGlue cloned at `~/opt/LightGlue`, installed editable.
- Carla 0.9.15 (Python 3.10 wheel from the LAC sim's bundled `wheelhouse/`).

**Configuration we made to the LAC sim (lives at `LAC_SIM/` inside this repo, gitignored — on the server it is a symlink to `/data/santiago/Lunar_Perception_Aware_Planning/LAC_SIM/`):**
- `RunLunarSimulator.sh`: plain native Linux wrapper — no WSL2/D3D12 env vars needed.
- `RunLeaderboard.sh`: `PATH` prepended with conda env bin (for `rerun`); `TEAM_CODE_ROOT` → this repo; `MISSIONS_SUBSET=1` (preset 2); `TEAM_AGENT` → `agents/nav_agent.py`.
- `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` — forces NVIDIA Vulkan ICD on the multi-GPU server.
- DEM naming fix: `LAC_SIM/results/Moon_Map_01_1_rep0.dat` present (copy of `data/DEMs/Moon_Map_01_2_rep0.dat`; agent reads results by MISSIONS_SUBSET index, not preset number).

**Compute:** `vader.stanford.edu` — native Ubuntu 20.04, 2× NVIDIA GeForce RTX 4090 (24 GB each), 125 GB RAM. **This is the primary sim + analysis machine.** WSL2 laptop is legacy (NVIDIA RTX 4060 + Intel Iris Xe on Windows 11); UE4 4.26 cannot run on WSL2 due to a `dzn` D3D12 fence-timeout bug (see Section 13.D).

**Data (current inventory):**
- `data/DEMs/Moon_Map_01_2_rep0.dat` — LAC preset 2 heightmap (180, 180, 4), delivered by TA Adam Dai. Channels `[x, y, z, rock_bool]`.
- `data/Example_Implementations/HW3_Final/data/dem_tile.npz` — LRO LOLA 5 m/pixel south-pole DEM tile (2000 × 2000 cells). The mid-scale evaluation environment.
- `data/Example_Implementations/HW3_Final/data/lac_data/` — 2000 stereo PNGs + `data_log.json` for LAC preset 2. The matched Phase 0 dataset.
- `data/Example_Implementations/HW3_Final/data/LAC/segmentation/` — 194 paired (FrontLeft, FrontLeft_semantic) frames for preset 7. The semantic-class palette is decoded in `scripts/train_segmentation.py:21-27` and is used for the rocks↔features auxiliary check.
- `models/unet_v2.pth` — 100 MB UNet++ segmentation weights from the JHU portal. Loaded by `lac/perception/segmentation.py:UnetSegmentation`.
- `models/` is gitignored.
- LAC image dataset: used for feature density predictor validation.
- Sun angle: fixed per mission, available from `self.config` JSON in the agent.

---

## 13. Implementation Order / Current Status

> **Date snapshot: 2026-05-26 (updated end-of-day).** Deadline: 2026-06-05 (report) / 2026-06-01 (slides). ~10 days remaining.
>
> **CRITICAL FRAMING UPDATE:** The original draft of this doc described a coverage-maximizing planner benchmarked against `gen_five_loops`/`spiral`/`triangles`. **That is no longer the project.** Santiago's chosen problem statement is now **goal-to-goal trajectory planning**: given a start pose and a goal pose, plan a path that minimizes localization uncertainty (measured by SLAM RMSE vs. ground truth) while still being efficient. The baseline is a slope-aware A* planner (the one from AA278 HW3 P4, also in `data/Example_Implementations/HW3_Final/`). Sections 9.1 and 9.3 of this doc still reference the old coverage-baseline framing — treat those as superseded by Section 13.A below.

### 13.A Two-environment evaluation (the actual paper plan)

- **Mid-scale**: LRO LOLA 5 m/pixel south-pole DEM tile at `data/Example_Implementations/HW3_Final/data/dem_tile.npz` (2000 × 2000 cells = 10 km × 10 km of real lunar terrain). Plan paths between hand-picked start/goal pairs on real lunar geometry. Metrics: predicted D-optimality, path length, max slope traversed. No SLAM here — this is the algorithmic + visual demo.
- **Small-scale**: LAC simulator on Moon_Map_01, picking presets with rich terrain. Hand-picked start/goal pairs (sim is only 27 m × 27 m). Run both the slope-aware A* baseline and our perception-aware planner; measure SLAM RMSE vs. ground truth, loop closure count, path length. This is the quantitative validation.

### 13.B Phase status

| Phase | Description | Status |
|---|---|---|
| **0**   | Pre-implementation feature-density validation (does roughness predict SuperPoint feature density?) | DONE — see Section 13.C |
| **0.5** | Same with shadow mask added (ρ_full = roughness × (1 − shadow_mask)) | DONE — see Section 13.C |
| **1**   | Pull HW3 `DEM` + `AStar` helpers into `lac/planning/` | **DONE** — `lac/planning/dem.py` + `lac/planning/astar.py`; added `DEM.from_lac_dat()` |
| **2**   | Build `PerceptionMap` (DEM + roughness + shadow-cast) | NOT STARTED — **NEXT** |
| **3**   | Build `EKFCovariancePropagator` (simplified 2D state) | NOT STARTED |
| **4**   | Build `PerceptionAwarePlanner` (A* with slope + perception terms) | NOT STARTED |
| **5**   | Build `PerceptionAwareAgent` (drop-in fork of `agents/nav_agent.py`) | NOT STARTED |
| **6**   | LRO mid-scale experiments (baseline vs ours) | NOT STARTED |
| **7**   | LAC small-scale experiments (RMSE measurement) | NOT STARTED |
| **8**   | LAC simulator bring-up | **DONE** — headless on `vader.stanford.edu` (native Linux + 2× RTX 4090). See Section 13.D. |
| **9**   | Paper writing | NOT STARTED |

Stretch A/B/C remain as described in Sections 8.5–8.7 but are only attempted if Phases 1–7 are clean.

### 13.C Phase 0 findings (key, condensed)

Validation script: `scripts/phase0_validation.py`. Inputs: `data/DEMs/Moon_Map_01_2_rep0.dat` (preset 2 LAC DEM) and 2000 stereo frames + ground-truth poses from preset 2 (`data/Example_Implementations/HW3_Final/data/lac_data/`). Detector: SuperPoint at `max_num_keypoints=2048` (same as the SLAM uses except for the cap raise to recover dynamic range).

**Three runs, preserved under `output/`:**
- `phase0_validation_kp512/` — initial 512-cap run, 91.9% saturation, results invalidated
- `phase0_validation_kp2048_no_shadow/` — uncapped roughness-only test
- `phase0_validation_az263_575/` — full ρ_full with sun az = 263.575° (astropy direct)
- `phase0_validation/` — full ρ_full with sun az = 83.575° (A/B-test flip; WORSE, so 263.575° is closer to correct)

**Headline correlations (n=2000 raw, n=39 IID-subsampled):**

| Hypothesis | Pearson_raw | Pearson_iid | Spearman | Interpretation |
|---|---|---|---|---|
| H1a roughness@pose → n_features | −0.42 | −0.22 | −0.42 | Strong, negative, robust |
| H1b roughness@LA=2 m → n_features | −0.11 | −0.19 | −0.16 | Weak negative |
| H2 rocks@LA → n_features | +0.07 | +0.16 | +0.14 | ~zero |
| H4a ρ_full@pose → n_features | −0.41 | −0.16 | −0.41 | Same as H1a (no improvement) |
| H4b ρ_full@LA=3 m → n_features | +0.05 | **+0.15** | +0.06 | Weak positive — *flipped sign* vs H1b |

**Conclusion: WEAK by the threshold heuristic, but DIRECTIONALLY CORRECT.** Adding the shadow mask flipped H4b from −0.15 to slightly-positive across all lookahead distances, exactly as the predictor design predicted. Magnitudes are too small to be useful, but the limiting factors are diagnosable:

1. **Preset 2 trajectory is geometrically uninteresting.** The rover stays in a 7 × 15 m sub-region with height range 1.86 m. Roughness varies only between 0.005 and 0.04 m for 95% of the run; the one big peak (0.30) lasts ~30 frames. There's no signal-to-correlate.
2. **Rover-cast shadow contaminates n_features.** In frames where the rover stands between sun and camera, the camera looks into the rover's own shadow → near-black image → SuperPoint finds little → low n_features that has nothing to do with terrain.
3. **Single-point lookahead under-samples the camera FOV.** The FrontLeft camera sees a band 4–10 m ahead at the surface; we sample ρ at a single cell at d ∈ {1, 1.5, 2, 3, 4} m. The 3–4 m endpoints showed the strongest H4b numbers but a forward-cone integration is the principled fix.
4. **n_features includes single-frame "hallucinated" features** that don't survive to SLAM matching. The real metric should count matched/tracked features across consecutive frames using LightGlue (see `lac/slam/feature_tracker.py:match_feats`), not raw SuperPoint count.
5. **Sun azimuth convention is approximately right** but not visually re-confirmed. The astropy value 263.575° (CCW from world +X) gave directionally-correct results; flipping to 83.575° made things worse. Final ground-truthing requires opening the sim, observing where rover/rock shadows actually fall, and back-solving the convention.

**Verdict: predictor concept is plausible; preset 2 is not a sufficient test bed. Re-validate after sim bring-up with a planned trajectory through rough/elevated regions, with matched-features as the response variable.** Do NOT pivot the predictor design based on preset 2 alone.

### 13.D Phase 8 status: sim bring-up — DEFINITIVELY BLOCKED on WSL2

**Full investigation log (2026-05-26):**

1. **Ubuntu 22.04 WSL**: UE4 4.26 dropped OpenGL → falls back to llvmpipe (CPU Vulkan) → GameThread timeout 60 s → crash. No fix possible without hardware Vulkan ICD.

2. **Ubuntu 24.04 WSL + kisak-mesa PPA**: Added `ppa:kisak/kisak-mesa`, got Mesa 26.1.1 with `dzn` (Microsoft Direct3D12 Vulkan implementation). `vkcube` confirmed rendering on NVIDIA RTX 4060. UE4 rejected the device with "Incompatible Vulkan driver found!" because `dzn` reports `conformanceVersion=0.0.0.0` and is missing `VK_EXT_robustness2`.

3. **Custom Vulkan layer** (`tools/lac_vulkan_compat_layer.c`): Wrote a layer that patches `conformanceVersion` to `1.3.0.0` and injects `VK_EXT_robustness2`. UE4 accepted the device and registered with the NVIDIA GPU. However, UE4 then hung indefinitely in `poll(/dev/dxg)` waiting for a D3D12 GPU fence that never signals. Threads show the main thread waiting on a D3D12 fence submitted during early pipeline state object (PSO) initialization; the GPU shows 1% utilization but never completes the work. This is a bug in `dzn`'s D3D12 command translation for UE4 4.26's PSO workload — not fixable via a Vulkan layer.

4. **LunarG Vulkan SDK devsim layer**: Not available for Ubuntu 24.04 Noble (LunarG only supports up to 22.04 Jammy).

**Verdict: WSL2 cannot run UE4 4.26.** The D3D12 fence hang is inside the Mesa `dzn` driver and cannot be patched externally. Native Linux + NVIDIA proprietary driver is required.

**Resolution: COMPLETE (2026-05-26).** Running headlessly on `vader.stanford.edu` — native Ubuntu 20.04, 2× NVIDIA GeForce RTX 4090 (24 GB each), 125 GB RAM. Simulator starts in ~5 s (PSO cache warm on fast GPU). `Step: 1, 2, 3, ...` confirmed; mission runs to completion and writes results files. See `docs/PROJECT_TURNOVER.md` Section 2 for the full server configuration and `docs/SIM_STARTUP.md` Section 0 for the setup procedure.

**Server-specific issues resolved during bring-up (see `SIM_STARTUP.md` Section 5 failures table and Section 0.11–0.13 for details):**
- `astropy` and `lunarsky` were missing from `requirements.txt` — must be installed separately.
- `rerun` binary not in PATH when using `setsid`/`nohup` — conda env bin must be prepended to PATH in `RunLeaderboard.sh`.
- Multi-GPU server has both NVIDIA and Intel Vulkan ICDs — must set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- Agent reads `results/Moon_Map_01_{MISSIONS_SUBSET}_rep0.dat` at startup — DEMs must be copied to `LAC_SIM/results/` with index-based names before first run.
- Processes launched from Claude Code / non-interactive shells must use `nohup setsid` to survive shell resets between commands.

### 13.E Updated step plan (current state 2026-05-26)

```
DONE ✅
├── 0. SSH server (vader.stanford.edu) fully operational — headless sim confirmed
├── 1. lac/planning/dem.py — DEM class with from_lac_dat() added
└── 2. lac/planning/astar.py — AStar[T] base class + path helpers

NEXT ← YOU ARE HERE
└── 2a. Build lac/planning/perception_map.py (PerceptionMap: DEM + roughness + shadow-cast)
     Does NOT require Phase 0 re-run — pure infrastructure over the settled DEM interface.

CAN RUN IN PARALLEL WITH (2a)
├── 3. Run baseline nav_agent.py end-to-end on one rich preset (presets 5/7/9 recommended)
├── 4. Visually verify sun azimuth convention in dem_overlays.png vs frame images
└── 5. Plan a trajectory that crosses flat/rough/elevated terrain

PHASE 0 RE-RUN (post-steps 3-5, before core planner)
├── 6. Re-run scripts/phase0_validation.py with rich trajectory + matched-features (LightGlue)
└── 7. Decision threshold: |H4b Pearson_iid| ≥ 0.2 before building the planner

CORE PLANNER (do NOT start until step 7 clears)
├── 8.  Build EKFCovariancePropagator (simplified 2D state per Section 8.2)
├── 9.  Build PerceptionAwarePlanner (subclass AStar; edge_cost = dist*(1+α*slope_term+β*(1-ρ)))
└── 10. Build PerceptionAwareAgent (drop-in fork of agents/nav_agent.py)

EXPERIMENTS
├── 11. LRO mid-scale: ours vs P4 baseline, multiple (start, goal) pairs, predicted-D-opt metric
├── 12. LAC small-scale: ours vs P4 baseline, hand-picked (start, goal) pairs, SLAM RMSE metric
└── 13. Ablations (β=0 = baseline, β sweep, shadow on/off)

PAPER (overlapping experiments)
├── 14. Section 5 figures (LRO paths + LAC RMSE + Phase 0 scatter)
└── 15. ION format, 6-8 pages, due 2026-06-05
```

Stretch A/B/C remain on the table but are explicitly NOT scheduled — they only enter the plan if step 10 finishes by day 8 of 10.

---

## 14. Critical "Do Not" List

- **Do not modify** `lac/slam/frontend.py` or `lac/slam/backend.py` — the SLAM stack is inherited as-is
- **Do not modify** `lac/planning/arc_planner.py` except for the single Stretch A perception term — and only after the core planner is working and validated
- **Do not modify** `agents/nav_agent.py` directly — create `agents/perception_aware_agent.py` as a fork
- **Do not** introduce a yaw sign flip. Positive yaw in the simulator is **clockwise**. The existing code already handles this.
- **Do not** use `MAP_SIZE` (180) as the scene size in meters. The map is 180 *cells* at 0.15 m/cell = 27 m extent. The scene is ±20 m. These are different numbers.
- **Do not** output velocity commands `(v, w)` from the global planner. The global planner outputs 2D `[x, y]` waypoints only. The ArcPlanner produces velocity commands.
- **Do not** over-engineer the feature density predictor. Start with DEM roughness + binary shadow mask. Add complexity only if the validation Pearson r is poor.
- **Do not** implement MPC for the loop closure incentive. The A* greedy γ term is the chosen approach. MPC is documented in Section 8.7 as a design decision for the paper — not as future implementation work.
- **Do not** implement Stretch B or C before the core planner is running and validated end-to-end in the simulator.

---

## 15. Quick Reference: Interface Summary

```python
# PerceptionMap
pm = PerceptionMap(heightmap, cell_width=0.15, sun_azimuth_deg=45.0, sun_elevation_deg=3.0)
rho  = pm.get_feature_density(x, y)           # float ∈ [0,1]
cost = pm.get_uncertainty_cost(x, y)           # float = 1 - rho
grid = pm.get_density_grid()                   # np.ndarray (180, 180)
pm.update_from_slam(semantic_points, pose)     # Stretch B: blend observed features into prior

# EKFCovariancePropagator
prop    = EKFCovariancePropagator()
score   = prop.propagate_path(path_xy, pm.get_feature_density)    # float (D-opt score)
P_final = prop.get_final_covariance(path_xy, pm.get_feature_density)  # (9,9) array

# LoopClosureMap (Stretch C)
lc_map = LoopClosureMap.from_trajectory(backend.get_trajectory(), pm)
lc_score = lc_map.get_lc_score(x, y)          # float ∈ [0,1]
lc_grid  = lc_map.get_lc_grid()               # np.ndarray (180, 180)

# PerceptionAwarePlanner (same interface as WaypointPlanner)
planner = PerceptionAwarePlanner(
    initial_pose, perception_map=pm,
    alpha=1.0, beta=2.0, gamma=0.0,           # gamma=0 until Stretch C
    loop_closure_map=None,                     # None until Stretch C
)
waypoint, advanced = planner.get_waypoint(step, pose, print_progress=True)
# waypoint: np.ndarray shape (2,) = [x, y] in global frame, or None if done
planner.replan(current_pose, updated_pm, updated_lc_map)  # Stretch B/C hook
```
