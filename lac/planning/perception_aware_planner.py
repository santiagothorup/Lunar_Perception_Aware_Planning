"""PerceptionAwarePlanner: goal-to-goal A* that prefers feature-rich, low-slope terrain.

Drop-in replacement for ``WaypointPlanner``: plans a path from a start to a goal whose edge cost
extends the AA278 HW3 P4 slope-aware A* with an additive perception term, then serves the path as
``[x, y]`` waypoints through the same ``get_waypoint`` interface ``NavAgent`` expects.

Edge cost (``n2`` is the cell being entered)::

    cost = dist * (1 + alpha * (slope / theta_ref)**slope_exp + beta * (1 - rho))

With ``beta = 0`` this is exactly the P4 baseline (clean ablation). ``rho`` is the PerceptionMap's
normalized feature density at ``n2``. The heuristic is straight-line distance (admissible: the
multiplier is always >= 1).

Coordinate handling: A* nodes are DEM ``(r, c)`` indices (``(rows = y, cols = x)``). The perception
map uses the opposite ``(x, y)`` layout, so ρ is always queried via **world** coordinates
(``dem.rc_to_xy`` -> ``perception_map.get_feature_density``) — the two grids never index each other.
"""
from __future__ import annotations

import numpy as np
from rich import print

from lac.planning.astar import AStar, GridCoord
from lac.planning.dem import DEM
from lac.planning.perception_map import PerceptionMap
from lac.planning.waypoint_planner import WAYPOINT_TIMEOUT
from lac.params import WAYPOINT_REACHED_DIST_THRESHOLD

# P4 baseline defaults (verified against AA278_2026_HW3_P4.ipynb).
THETA_MAX_DEG = 20.0   # maximum traversable slope
THETA_REF_DEG = 10.0   # reference slope where the penalty becomes significant
SLOPE_EXPONENT = 2.0   # penalty aggressiveness
ALPHA = 2.0            # slope-penalty weight
BETA = 2.0             # perception (uncertainty) weight; beta=0 -> exact P4 baseline

# Look-ahead distances (m) at which the perception reward samples rho the camera would see ahead.
# A short forward band (Phase 0's strongest look-ahead correlation spanned ~1.5-4 m, and single-point
# look-ahead under-samples the camera FOV) -- averaged for robustness; tunable via config.
LOOKAHEAD_DISTANCES_M = (1.5, 2.5, 3.5)

# Traversability penalty weight: discourages routing onto rough/rocky cells (which stall the local
# ArcPlanner and degrade SLAM). 0 = off. Pairs with beta so the planner prefers SMOOTH ground that
# overlooks feature-rich terrain rather than driving into it.
TRAV_WEIGHT = 0.0

# v1b: weight on the global anchor-proximity reward (lowers edge cost near precomputed high-rho
# anchor hubs so paths revisit them when topology allows -- the lever for loop-closure-driven
# localization). 0 = off (planner reduces to baseline + look-ahead-rho + trav).
ANCHOR_WEIGHT = 0.0
ANCHOR_RADIUS_M = 3.0

# Q4 (shadow-aware A*): penalty per shadowed cell. Used by the "shadow variant" planner that drops
# the rho/trav terms entirely and routes via slope + shadow only. 0 = off (no shadow penalty).
SHADOW_WEIGHT = 0.0

# v1c: online detour defaults. Trigger fires when steps_since_lc exceeds the threshold; detour aborts
# after MAX_DETOUR_STEPS without a verified LC at the target anchor; at most MAX_DETOUR_ATTEMPTS_PER
# _LEG detours are attempted per sub-goal leg (deadlock defense). LC_DIST_THRESHOLD_M mirrors the
# backend's config (loop_closure.distance_threshold_m); WAYPOINT_TIMEOUT_DETOUR widens the stuck
# detection during detours so ArcPlanner deflections don't trigger forced advances mid-detour.
STEPS_SINCE_LC_THRESHOLD = 400
MAX_DETOUR_STEPS = 1500
MAX_DETOUR_ATTEMPTS_PER_LEG = 3
LC_DIST_THRESHOLD_M = 1.0
WAYPOINT_TIMEOUT_DETOUR = 4000


class PlannerError(Exception):
    """Raised when the planner cannot construct a feasible plan."""


def resample_path_xy_with_keepers(
    path_xy: np.ndarray, spacing: float, keepers=()
) -> tuple[np.ndarray, list[int]]:
    """Resample a dense polyline to ~uniform ``spacing`` (keeps the endpoints).

    For each ``(x, y)`` in ``keepers``, also return the index into the resampled path that is closest
    in arc-length to that vertex -- so callers can mark sub-goal vertices (or detour endpoints) and
    detect when ``waypoint_idx`` crosses them. The resampled path itself is unchanged by keepers
    (no force-insertion) so single-goal callers with ``keepers=[goal]`` get bit-identical output.
    """
    if len(path_xy) < 2:
        return path_xy.copy(), [0] * len(list(keepers))
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return path_xy[:1].copy(), [0] * len(list(keepers))
    n = max(1, int(np.ceil(total / spacing)))
    s_new = np.linspace(0.0, total, n + 1)
    resampled = np.column_stack(
        [np.interp(s_new, s, path_xy[:, 0]), np.interp(s_new, s, path_xy[:, 1])]
    )
    keeper_indices: list[int] = []
    for k in keepers:
        k_xy = np.asarray(k, dtype=np.float64)[:2]
        i_path = int(np.argmin(np.linalg.norm(path_xy - k_xy, axis=1)))
        keeper_indices.append(int(np.argmin(np.abs(s_new - s[i_path]))))
    return resampled, keeper_indices


class PerceptionAwarePlanner(AStar[GridCoord]):
    def __init__(
        self,
        start_xy,
        goal_sequence,
        perception_map: PerceptionMap,
        dem: DEM,
        *,
        alpha: float = ALPHA,
        beta: float = BETA,
        theta_ref_deg: float = THETA_REF_DEG,
        slope_exp: float = SLOPE_EXPONENT,
        theta_max_deg: float = THETA_MAX_DEG,
        lookahead_distances=LOOKAHEAD_DISTANCES_M,
        trav_weight: float = TRAV_WEIGHT,
        anchor_weight: float = ANCHOR_WEIGHT,
        anchor_radius_m: float = ANCHOR_RADIUS_M,
        shadow_weight: float = SHADOW_WEIGHT,
        detour_online_enabled: bool = False,
        detour_target_mode: str = "anchor",
        detour_random_target: bool = False,  # back-compat alias for detour_target_mode="random"
        detour_keyframe_min_m: float = 2.0,
        detour_keyframe_max_m: float = 8.0,
        steps_since_lc_threshold: int = STEPS_SINCE_LC_THRESHOLD,
        max_detour_steps: int = MAX_DETOUR_STEPS,
        max_detour_attempts_per_leg: int = MAX_DETOUR_ATTEMPTS_PER_LEG,
        lc_dist_threshold_m: float = LC_DIST_THRESHOLD_M,
        waypoint_timeout_detour_steps: int = WAYPOINT_TIMEOUT_DETOUR,
        waypoint_spacing: float = 2.0,
        waypoint_reached_threshold: float = WAYPOINT_REACHED_DIST_THRESHOLD,
    ):
        self.dem = dem
        self.perception_map = perception_map
        self.alpha = alpha
        self.beta = beta
        self.theta_ref_deg = theta_ref_deg
        self.slope_exp = slope_exp
        self.theta_max_deg = theta_max_deg
        self.lookahead_distances = tuple(lookahead_distances)
        self.trav_weight = trav_weight
        self.anchor_weight = anchor_weight
        self.anchor_radius_m = anchor_radius_m
        self.shadow_weight = float(shadow_weight)
        # v1b: anchors come from perception_map.anchors_xy (must be precomputed by the agent via
        # PerceptionMap.compute_anchors before constructing the planner). None / empty -> reward = 0.
        a = getattr(perception_map, "anchors_xy", None)
        self._anchors_xy: np.ndarray | None = (
            np.asarray(a, dtype=np.float64) if a is not None and len(a) > 0 else None
        )
        self.waypoint_spacing = waypoint_spacing
        self.waypoint_reached_threshold = waypoint_reached_threshold

        # v1c/v1d: online detour state. detour_online_enabled gates the maybe_detour call in
        # get_waypoint; detour_target_mode selects how the detour endpoint is chosen:
        #   "anchor"   (v1c default): pick from precomputed PerceptionMap.anchors_xy.
        #   "keyframe" (v1d): pick an LC-eligible past keyframe (guaranteed driveable + LC-firing,
        #                     since the rover already visited it).
        #   "random"   (control): pick a random slope-feasible non-anchor cell.
        # detour_random_target is the v1c back-compat alias for mode="random".
        mode = str(detour_target_mode).lower()
        if detour_random_target:
            mode = "random"
        if mode not in ("anchor", "keyframe", "random"):
            raise ValueError(f"detour_target_mode must be anchor/keyframe/random; got {mode!r}")
        self.detour_online_enabled = bool(detour_online_enabled)
        self.detour_target_mode = mode
        self.detour_random_target = (mode == "random")  # kept for log strings / sidecar
        self.detour_keyframe_min_m = float(detour_keyframe_min_m)
        self.detour_keyframe_max_m = float(detour_keyframe_max_m)
        self.steps_since_lc_threshold = int(steps_since_lc_threshold)
        self.max_detour_steps = int(max_detour_steps)
        self.max_detour_attempts_per_leg = int(max_detour_attempts_per_leg)
        self.lc_dist_threshold_m = float(lc_dist_threshold_m)
        self.waypoint_timeout_detour_steps = int(waypoint_timeout_detour_steps)
        self._slam_provider = None  # set via set_slam_provider() by the agent
        self._detour_active = False
        self._detour_target_xy: np.ndarray | None = None
        self._detour_start_step = 0
        self._detour_baseline_lc_count = 0
        self._detour_history: list[np.ndarray] = []
        self._detour_attempts_this_leg = 0
        self._last_subgoal_idx_seen = 0  # to reset per-leg attempt cap when sub-goal advances
        # v1c-fix: _subgoal_index_offset = global sub-goal index of the FIRST entry currently in
        # _subgoal_waypoint_indices. Always 0 for the original plan; becomes _current_subgoal_idx
        # after a detour splice (the new plan starts at the next-to-reach sub-goal). The advance
        # check in get_waypoint indexes _subgoal_waypoint_indices by (_current_subgoal_idx - offset).
        self._subgoal_index_offset = 0
        self._detour_rng = np.random.default_rng(0)  # for random_detour control reproducibility
        # Counters for diagnostics / logging.
        self.detour_attempts = 0
        self.detour_successes = 0
        self.detour_events: list[dict] = []  # one dict per attempt: {step, target_xy, success}

        self.start_xy = np.asarray(start_xy, dtype=np.float64)[:2]
        # goal_sequence: (M, 2). Back-compat: a bare (2,) point is wrapped to a 1-leg sequence.
        gs = np.asarray(goal_sequence, dtype=np.float64)
        if gs.ndim == 1 and gs.shape == (2,):
            gs = gs.reshape(1, 2)
        if gs.ndim != 2 or gs.shape[1] != 2 or gs.shape[0] < 1:
            raise PlannerError(f"goal_sequence must be (M, 2) with M>=1; got shape {gs.shape}")
        self._goal_sequence = gs
        self.goal_xy = gs[-1].copy()  # back-compat: last sub-goal is "the" goal for logging
        self._current_subgoal_idx = 0

        # Plan once: per-leg A* stitched into one dense path; subgoal_path_indices marks each
        # sub-goal vertex's index in the concatenated dense path (consumed by the resampler keepers).
        self.path_xy, subgoal_path_indices = self._plan_sequence()
        sub_xy = [self.path_xy[i] for i in subgoal_path_indices]
        waypoints, keeper_indices = resample_path_xy_with_keepers(
            self.path_xy, self.waypoint_spacing, keepers=sub_xy
        )
        # Drop the leading point (the start): the rover begins there, so the first *target* must be
        # ahead of it. This mirrors WaypointPlanner (whose patterns never put a waypoint on the
        # spawn) and, crucially, prevents an immediate waypoint-advance at step 1 -- which would
        # trigger the inherited NavAgent.update_map() -> backend.project_point_map() before the SLAM
        # backend has any points (np.vstack on an empty list).
        if len(waypoints) > 1:
            self.waypoints = waypoints[1:]
            self._subgoal_waypoint_indices = [max(0, i - 1) for i in keeper_indices]
        else:
            self.waypoints = waypoints
            self._subgoal_waypoint_indices = list(keeper_indices)
        self.waypoint_idx = 0
        self.last_waypoint_step = 0

    # ------------------------------------------------------------------ #
    #  AStar hooks
    # ------------------------------------------------------------------ #

    def _cell_slope_deg(self, r: int, c: int) -> float:
        """Terrain slope (deg) at DEM cell (r, c); inf if out of bounds or undefined."""
        if not self.dem.rc_in_bounds(r, c):
            return float("inf")
        gx, gy = self.dem.gx[r, c], self.dem.gy[r, c]
        if not (np.isfinite(gx) and np.isfinite(gy)):
            return float("inf")
        return float(np.degrees(np.arctan(np.hypot(gx, gy))))

    def neighbors(self, node: GridCoord):
        r, c = node
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not self.dem.rc_in_bounds(nr, nc):
                    continue
                if self._cell_slope_deg(nr, nc) > self.theta_max_deg:
                    continue
                out.append((nr, nc))
        return out

    def _anchor_proximity_reward(self, x: float, y: float) -> float:
        """Reward in [0, 1] for being within ``anchor_radius_m`` of any anchor (linear falloff)."""
        if self._anchors_xy is None:
            return 0.0
        dx = self._anchors_xy[:, 0] - x
        dy = self._anchors_xy[:, 1] - y
        d_min = float(np.sqrt(np.min(dx * dx + dy * dy)))
        reward = max(0.0, 1.0 - d_min / self.anchor_radius_m)
        # Bound assertion (audit fix): min-distance form guarantees reward in [0, 1] -> the cost
        # multiplier stays >= 1 -> Euclidean heuristic remains admissible.
        assert 0.0 <= reward <= 1.0
        return reward

    def distance_between(self, n1: GridCoord, n2: GridCoord) -> float:
        x1, y1 = self.dem.rc_to_xy(*n1)
        x2, y2 = self.dem.rc_to_xy(*n2)
        dist = float(np.hypot(x2 - x1, y2 - y1))
        slope = self._cell_slope_deg(*n2)
        # Reward the feature density the camera SEES from n2 looking along the direction of travel
        # (look-ahead rho), not the roughness underfoot -- matches the Phase 0 predictor and keeps
        # the rover on traversable ground rather than routing it onto rough/rocky cells.
        rho = self.perception_map.get_lookahead_density(
            x2, y2, (x2 - x1, y2 - y1), self.lookahead_distances
        )
        # Traversability penalty on the entered cell: prefer smooth ground (keeps the rover off
        # rough/rocky terrain that stalls the ArcPlanner and degrades SLAM, while beta still rewards
        # SEEING feature-rich terrain ahead via the look-ahead rho above).
        trav = self.perception_map.get_traversability_cost(x2, y2)
        # v1b: anchor-proximity reward -- routes the path through high-rho hubs so they can serve as
        # loop-closure targets. (1 - reward) is in [0, 1], so the multiplier stays >= 1.
        anchor_pen = 1.0 - self._anchor_proximity_reward(x2, y2)
        # Q4: shadow penalty -- the planner pays shadow_weight * 1.0 per shadowed cell entered, 0
        # per sunlit cell. Used by the "shadow variant" (beta=0, trav=0) to keep the rover in
        # sunlit (feature-visible) terrain WITHOUT rewarding rough cells.
        shadow = self.perception_map.get_shadow_value(x2, y2)
        return dist * (
            1.0
            + self.alpha * (slope / self.theta_ref_deg) ** self.slope_exp
            + self.beta * (1.0 - rho)
            + self.trav_weight * trav
            + self.anchor_weight * anchor_pen
            + self.shadow_weight * shadow
        )

    def heuristic_cost_estimate(self, current: GridCoord, goal: GridCoord) -> float:
        x1, y1 = self.dem.rc_to_xy(*current)
        x2, y2 = self.dem.rc_to_xy(*goal)
        return float(np.hypot(x2 - x1, y2 - y1))

    # ------------------------------------------------------------------ #
    #  Planning
    # ------------------------------------------------------------------ #

    def _astar_xy(self, start_xy, goal_xy) -> np.ndarray:
        """Run A* between two world points; return the dense (N, 2) path or raise PlannerError."""
        s_xy = np.asarray(start_xy, dtype=np.float64)[:2]
        g_xy = np.asarray(goal_xy, dtype=np.float64)[:2]
        start_rc = self.dem.xy_to_rc(*s_xy)
        goal_rc = self.dem.xy_to_rc(*g_xy)
        node_path = self.astar(start_rc, goal_rc)
        if node_path is None:
            raise PlannerError(
                f"No slope-feasible path from {s_xy.tolist()} to {g_xy.tolist()} "
                f"(theta_max={self.theta_max_deg} deg)."
            )
        return np.array([self.dem.rc_to_xy(r, c) for r, c in node_path], dtype=np.float64)

    def _plan_sequence(self) -> tuple[np.ndarray, list[int]]:
        """Plan A* per sub-goal leg, concatenate; return (dense path, sub-goal indices in path).

        Sub-goal indices mark where each sub-goal vertex sits in the concatenated dense path so the
        resampler keepers can locate them for sub-goal-advance tracking in get_waypoint.
        """
        legs: list[np.ndarray] = []
        subgoal_indices: list[int] = []
        cur = self.start_xy
        cum_len = 0
        for i, goal in enumerate(self._goal_sequence):
            try:
                leg = self._astar_xy(cur, goal)
            except PlannerError as e:
                raise PlannerError(
                    f"v1a sub-goal {i} unreachable (from {cur.tolist()} to {goal.tolist()}): {e}"
                ) from e
            if legs:
                leg = leg[1:]  # drop the duplicated leg-junction vertex
            legs.append(leg)
            cum_len += len(leg)
            subgoal_indices.append(cum_len - 1)
            cur = np.asarray(goal, dtype=np.float64)
        return np.concatenate(legs, axis=0), subgoal_indices

    # ------------------------------------------------------------------ #
    #  v1c: online detour (reactive replan when SLAM has been drifting without an LC)
    # ------------------------------------------------------------------ #

    def set_slam_provider(self, fn):
        """Inject a callable returning ``{"pose_idx", "loop_closures", "keyframe_xy_lc_eligible",
        "keyframe_xy_all"}`` -- the agent supplies this from its backend (see perception_aware_agent
        ._slam_state). Without a provider, maybe_detour is a no-op."""
        self._slam_provider = fn

    def _select_detour_anchor(self, pose_xy: np.ndarray, kf_eligible: np.ndarray):
        """Pick the anchor that (a) is not in _detour_history, (b) lies within 1.5x LC distance of
        an LC-eligible keyframe, and (c) maximizes rho/(1 + detour_extra_distance). Returns
        ``(x, y)`` or None if no candidate qualifies."""
        if self._anchors_xy is None or len(self._anchors_xy) == 0 or len(kf_eligible) == 0:
            return None
        next_subgoal_xy = self._goal_sequence[
            min(self._current_subgoal_idx, len(self._goal_sequence) - 1)
        ]
        d_thresh = 1.5 * self.lc_dist_threshold_m
        best, best_score = None, -np.inf
        for i, a in enumerate(self._anchors_xy):
            if any(np.linalg.norm(a - h) < 1e-6 for h in self._detour_history):
                continue
            d_kf = np.linalg.norm(kf_eligible - a, axis=1).min()
            if d_kf > d_thresh:
                continue
            # detour_extra_distance: (pose -> a -> next_subgoal) minus direct (pose -> next_subgoal).
            d_direct = float(np.linalg.norm(next_subgoal_xy - pose_xy))
            d_via = float(np.linalg.norm(a - pose_xy) + np.linalg.norm(next_subgoal_xy - a))
            extra = max(0.0, d_via - d_direct)
            score = (
                float(self.anchors_score_at(i)) / (1.0 + extra)
                if self.anchors_score_at(i) is not None
                else 1.0 / (1.0 + extra)
            )
            if score > best_score:
                best_score, best = score, a
        return best if best is None else np.asarray(best, dtype=np.float64)

    def anchors_score_at(self, i: int):
        """Helper: return the precomputed anchor score (rho_norm) at index i, or None."""
        s = getattr(self.perception_map, "anchors_score", None)
        if s is None or i >= len(s):
            return None
        return float(s[i])

    def _select_detour_keyframe(self, pose_xy: np.ndarray, kf_eligible: np.ndarray):
        """v1d: pick a past LC-eligible keyframe as the detour target.

        Past keyframes are by construction driveable (the rover visited them) and revisiting one
        within LC_DIST_THRESHOLD will fire an LC by the backend's geometric trigger. We restrict to
        keyframes ``detour_keyframe_min_m <= dist_to_pose <= detour_keyframe_max_m`` so the detour
        is non-trivial but bounded, and skip recent-failure history. Score: ``1/(1+extra_dist)``.
        """
        if kf_eligible is None or len(kf_eligible) == 0:
            return None
        next_subgoal_xy = self._goal_sequence[
            min(self._current_subgoal_idx, len(self._goal_sequence) - 1)
        ]
        d_pose = np.linalg.norm(kf_eligible - pose_xy, axis=1)
        mask = (d_pose >= self.detour_keyframe_min_m) & (d_pose <= self.detour_keyframe_max_m)
        candidates = kf_eligible[mask]
        if len(candidates) == 0:
            return None
        d_direct = float(np.linalg.norm(next_subgoal_xy - pose_xy))
        best, best_score = None, -np.inf
        for kf in candidates:
            if any(np.linalg.norm(kf - h) < 1e-6 for h in self._detour_history):
                continue
            d_via = float(np.linalg.norm(kf - pose_xy) + np.linalg.norm(next_subgoal_xy - kf))
            extra = max(0.0, d_via - d_direct)
            score = 1.0 / (1.0 + extra)
            if score > best_score:
                best_score, best = score, kf
        return None if best is None else np.asarray(best, dtype=np.float64)

    def _select_random_detour_target(self, pose_xy: np.ndarray):
        """Control: pick a slope-feasible cell roughly ``2 * waypoint_spacing`` from current pose in
        a random direction so the detour cadence matches the anchor variant, but the target is NOT
        an anchor (rules out 'any extra revisit helps')."""
        for _ in range(40):
            theta = float(self._detour_rng.uniform(0.0, 2.0 * np.pi))
            dist = float(self._detour_rng.uniform(2.0, 5.0))
            tgt = np.array([pose_xy[0] + dist * np.cos(theta), pose_xy[1] + dist * np.sin(theta)])
            r, c = self.dem.xy_to_rc(*tgt)
            if not self.dem.rc_in_bounds(r, c):
                continue
            if self._cell_slope_deg(r, c) > self.theta_max_deg:
                continue
            return tgt
        return None

    def _splice_detour(self, pose_xy: np.ndarray, target_xy: np.ndarray, step: int) -> bool:
        """Plan ``pose -> target -> remaining sub-goals``; replace ``self.waypoints`` wholesale.

        The detour leg goes ``pose -> target``, then a chain of A* legs visits every sub-goal from
        ``_current_subgoal_idx`` through the end of ``_goal_sequence`` (fix for the v1c bug that
        dropped post-next-subgoal sub-goals after a splice and ended the mission early). Re-baselines
        ``_subgoal_waypoint_indices`` for the new plan and sets ``_subgoal_index_offset`` so the
        global sub-goal counter survives the re-plan. Returns True iff the new list was installed.
        """
        remaining = [np.asarray(g, dtype=np.float64) for g in
                     self._goal_sequence[self._current_subgoal_idx:]]
        if not remaining:
            return False
        target = np.asarray(target_xy, dtype=np.float64)
        try:
            legs = [self._astar_xy(pose_xy, target)]
            prev = target
            for g in remaining:
                leg = self._astar_xy(prev, g)
                legs.append(leg[1:])  # drop duplicated joining vertex
                prev = g
        except PlannerError:
            return False
        new_tail = np.concatenate(legs, axis=0)
        if len(new_tail) < 2:
            return False
        # Keepers: detour target first, then every remaining sub-goal (in plan order).
        keepers = [target] + remaining
        new_wps, keeper_indices = resample_path_xy_with_keepers(
            new_tail, self.waypoint_spacing, keepers=keepers
        )
        if len(new_wps) <= 1:
            return False
        # Drop the first sample (lies near current pose -> avoids instant advance).
        self.waypoints = new_wps[1:]
        # Shift keeper indices for the dropped first waypoint; floor at 0.
        shifted = [max(0, i - 1) for i in keeper_indices]
        # First keeper is the detour target (not a global sub-goal). Skip it; the rest map 1:1 to
        # _goal_sequence[_current_subgoal_idx:].
        self._subgoal_waypoint_indices = shifted[1:]
        self._subgoal_index_offset = self._current_subgoal_idx
        self.waypoint_idx = 0
        self.last_waypoint_step = step
        return True

    def maybe_detour(self, step: int, pose: np.ndarray) -> bool:
        """If we've been drifting too long without an LC and a valid target exists, splice a detour.

        Returns True iff a new detour was installed THIS call (informational; the planner does not
        require the caller to act on it). Called from inside ``get_waypoint`` so the new waypoints
        are visible to the rover the same step.
        """
        if not self.detour_online_enabled or self._slam_provider is None:
            return False
        # Reset per-leg attempt cap when sub-goal advances.
        if self._current_subgoal_idx != self._last_subgoal_idx_seen:
            self._detour_attempts_this_leg = 0
            self._last_subgoal_idx_seen = self._current_subgoal_idx
        s = self._slam_provider()
        lc_list = s.get("loop_closures") or []
        pose_xy = pose[:2, 3]
        # Active detour: clear only on VERIFIED LC at the target anchor, or after a hard step cap.
        if self._detour_active:
            new_lcs = len(lc_list) - self._detour_baseline_lc_count
            cleared = False
            if new_lcs > 0:
                anchor_kf_idx = int(lc_list[-1][0])
                all_kf = s.get("keyframe_xy_all")
                if all_kf is not None and 0 <= anchor_kf_idx < len(all_kf):
                    anchor_kf_xy = np.asarray(all_kf[anchor_kf_idx], dtype=np.float64)
                    if np.linalg.norm(anchor_kf_xy - self._detour_target_xy) <= 2.0 * self.lc_dist_threshold_m:
                        self._detour_active = False
                        self.detour_successes += 1
                        if self.detour_events:
                            self.detour_events[-1]["success"] = True
                            self.detour_events[-1]["clear_step"] = step
                        cleared = True
            if not cleared and step - self._detour_start_step > self.max_detour_steps:
                self._detour_active = False
                self._detour_history.append(self._detour_target_xy)
                if self.detour_events:
                    self.detour_events[-1]["clear_step"] = step
            return False

        if self._detour_attempts_this_leg >= self.max_detour_attempts_per_leg:
            return False
        last_lc_step = int(lc_list[-1][1]) if lc_list else 0
        pose_idx = int(s.get("pose_idx", 0))
        if pose_idx - last_lc_step < self.steps_since_lc_threshold:
            return False

        kf_eligible = s.get("keyframe_xy_lc_eligible")
        kf_eligible = np.asarray(kf_eligible, dtype=np.float64) if kf_eligible is not None else np.empty((0, 2))
        if self.detour_target_mode == "random":
            target = self._select_random_detour_target(pose_xy)
        elif self.detour_target_mode == "keyframe":
            target = self._select_detour_keyframe(pose_xy, kf_eligible)
        else:  # "anchor"
            target = self._select_detour_anchor(pose_xy, kf_eligible)
        if target is None:
            return False
        if not self._splice_detour(pose_xy, target, step):
            return False
        self._detour_active = True
        self._detour_target_xy = np.asarray(target, dtype=np.float64)
        self._detour_baseline_lc_count = len(lc_list)
        self._detour_start_step = step
        self._detour_attempts_this_leg += 1
        self.detour_attempts += 1
        self.detour_events.append(
            {"step": step, "target_xy": self._detour_target_xy.tolist(), "success": False}
        )
        print(
            f"[bold magenta]DETOUR #{self.detour_attempts} -> target "
            f"({self._detour_target_xy[0]:+.2f}, {self._detour_target_xy[1]:+.2f}) at step {step}"
        )
        return True

    # ------------------------------------------------------------------ #
    #  Waypoint serving (interface identical to WaypointPlanner.get_waypoint)
    # ------------------------------------------------------------------ #

    def get_waypoint(
        self, step: int, pose: np.ndarray, print_progress: bool = False
    ) -> tuple[np.ndarray | None, bool]:
        """Return ``(waypoint_xy (2,), advanced)`` or ``(None, True)`` when the goal is reached."""
        # v1c: online detour first -- if a new detour is spliced in, waypoints/waypoint_idx update
        # so the rover immediately starts driving toward the detour target.
        self.maybe_detour(step, pose)
        advanced = False
        waypoint = self.waypoints[self.waypoint_idx]
        xy_position = pose[:2, 3]

        # v1c: widen the stuck-detection timeout while a detour is active (ArcPlanner deflections
        # around rocks near anchors can otherwise force an unwanted advance).
        timeout = self.waypoint_timeout_detour_steps if self._detour_active else WAYPOINT_TIMEOUT
        if step - self.last_waypoint_step > timeout:
            print(f"[bold red]WAYPOINT TIMEOUT ON {self.waypoint_idx + 1}/{len(self.waypoints)}")
            advanced = True

        distance = np.linalg.norm(xy_position - waypoint)
        if distance < self.waypoint_reached_threshold:
            advanced = True

        if advanced:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):  # goal reached
                self._current_subgoal_idx = len(self._goal_sequence)  # final sub-goal consumed
                self.waypoint_idx = 0
                return None, True
            waypoint = self.waypoints[self.waypoint_idx]
            self.last_waypoint_step = step
            # Increment sub-goal counter when we cross the next-to-reach sub-goal's waypoint index.
            # _subgoal_index_offset shifts after a detour splice; entry [k - offset] holds the
            # waypoint-index for global sub-goal k.
            while True:
                local = self._current_subgoal_idx - self._subgoal_index_offset
                if not (0 <= local < len(self._subgoal_waypoint_indices)):
                    break
                if self.waypoint_idx <= self._subgoal_waypoint_indices[local]:
                    break
                if print_progress:
                    print(
                        f"[bold cyan]Sub-goal {self._current_subgoal_idx + 1}/"
                        f"{len(self._goal_sequence)} reached"
                    )
                self._current_subgoal_idx += 1

        if print_progress:
            print(f"Waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}: {waypoint}")
            print(f"Distance to waypoint: {distance:.2f} m")

        return waypoint, advanced
