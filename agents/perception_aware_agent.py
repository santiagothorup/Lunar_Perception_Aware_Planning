#!/usr/bin/env python
"""PerceptionAwareAgent: a NavAgent fork that plans goal-to-goal with PerceptionAwarePlanner.

Identical to ``NavAgent`` except ``setup()`` swaps the coverage ``WaypointPlanner`` for a
``PerceptionAwarePlanner`` that routes start -> goal through feature-rich, low-slope terrain. Every
other behavior — SLAM frontend/backend, ArcPlanner steering, the ``get_waypoint`` -> ``plan_arc``
control loop, logging, and RMSE-vs-ground-truth in ``finalize()`` — is inherited unchanged, so
``nav_agent.py`` is not modified.
"""
import json
import os

import numpy as np
from rich import print

# The leaderboard loads agents by inserting this directory into sys.path and importing the bare
# module name (agent_wrapper._get_agent_instance), so NavAgent is imported as a sibling by bare name.
from nav_agent import NavAgent
from nav_agent import get_entry_point as _nav_get_entry_point  # for output-dir path resolution
from lac.planning.dem import DEM
from lac.planning.perception_map import PerceptionMap
from lac.planning.perception_aware_planner import PerceptionAwarePlanner


def get_entry_point():
    return "PerceptionAwareAgent"


class PerceptionAwareAgent(NavAgent):
    def setup(self, path_to_conf_file):
        # Build the full NavAgent stack (cameras, SLAM, arc planner, logging, ground-truth map).
        # This constructs a placeholder WaypointPlanner from config["planning"]["trajectory"]
        # (use "perception_aware" -> a single no-op waypoint); it is replaced below by the real
        # start->goal PerceptionAwarePlanner, so it never drives.
        super().setup(path_to_conf_file)

        # Ground-truth DEM for the preset; written to results/ by every run (cwd is LAC_SIM here,
        # matching NavAgent's own ground_truth_map load). preset = MISSIONS_SUBSET index.
        preset = os.environ.get("MISSIONS_SUBSET")
        dem_path = f"results/Moon_Map_01_{preset}_rep0.dat"

        plan_cfg = self.config["planning"]
        pmap_cfg = self.config.get("perception_map", {})

        dem = DEM.from_lac_dat(dem_path)
        perception_map = PerceptionMap.from_lac_dat(
            dem_path,
            sun_azimuth_deg=pmap_cfg.get("sun_azimuth_deg", 263.575),
            sun_altitude_deg=pmap_cfg.get("sun_altitude_deg", 1.488),
            roughness_window=pmap_cfg.get("roughness_window", 5),
        )
        # v1b: precompute anchor hubs (high rho, slope-feasible, rock-sparse). Always run so anchors
        # are available for diagnostics even when anchor_weight=0; planner only routes through them
        # when anchor_weight > 0.
        anchors = perception_map.compute_anchors(
            dem,
            min_separation_m=plan_cfg.get("anchor_min_separation_m", 4.0),
            max_count=plan_cfg.get("anchor_max_count", 12),
            min_rho=plan_cfg.get("anchor_min_rho", 0.4),
            rock_density_max=plan_cfg.get("anchor_rock_density_max", 0.05),
            rock_window_m=plan_cfg.get("anchor_rock_window_m", 1.0),
            roughness_max=plan_cfg.get("anchor_roughness_max", 1.0),
        )
        print(f"[bold cyan]PerceptionMap: {len(anchors)} anchors extracted")

        # start_xy defaults to the mission spawn; set start_xy + goal_xy/goal_sequence for routed runs.
        start_xy = plan_cfg.get("start_xy") or self.initial_pose[:2, 3]
        # v1a: prefer goal_sequence (ordered list of (x, y)); fall back to single goal_xy for compat.
        goal_sequence = plan_cfg.get("goal_sequence")
        if goal_sequence is None:
            goal_sequence = [plan_cfg["goal_xy"]]
        lc_cfg = self.config.get("loop_closure", {})
        self.planner = PerceptionAwarePlanner(
            np.asarray(start_xy, dtype=float),
            np.asarray(goal_sequence, dtype=float),
            perception_map,
            dem,
            alpha=plan_cfg.get("alpha", 2.0),
            beta=plan_cfg.get("beta", 2.0),
            lookahead_distances=tuple(plan_cfg.get("perception_lookahead_m", (1.5, 2.5, 3.5))),
            trav_weight=plan_cfg.get("traversability_weight", 0.0),
            anchor_weight=plan_cfg.get("anchor_weight", 0.0),
            anchor_radius_m=plan_cfg.get("anchor_radius_m", 3.0),
            shadow_weight=plan_cfg.get("shadow_weight", 0.0),
            detour_online_enabled=plan_cfg.get("detour_online_enabled", False),
            detour_target_mode=plan_cfg.get("detour_target_mode", "anchor"),
            detour_random_target=plan_cfg.get("detour_random_target", False),
            detour_keyframe_min_m=plan_cfg.get("detour_keyframe_min_m", 2.0),
            detour_keyframe_max_m=plan_cfg.get("detour_keyframe_max_m", 8.0),
            steps_since_lc_threshold=plan_cfg.get("steps_since_lc_threshold", 400),
            max_detour_steps=plan_cfg.get("max_detour_steps", 1500),
            max_detour_attempts_per_leg=plan_cfg.get("max_detour_attempts_per_leg", 3),
            lc_dist_threshold_m=lc_cfg.get("distance_threshold_m", 1.0),
            waypoint_timeout_detour_steps=plan_cfg.get("waypoint_timeout_detour_steps", 4000),
            waypoint_spacing=plan_cfg.get("waypoint_spacing_m", 2.0),
            waypoint_reached_threshold=plan_cfg["waypoint_reached_threshold_m"],
        )
        # v1c: SLAM-state provider closure (planner polls it each step via maybe_detour).
        # keyframe_xy_lc_eligible excludes the last LOOP_CLOSURE_EXCLUDE keyframes (backend won't
        # match against them), so anchors near only-recent keyframes are skipped.
        from lac.slam.backend import LOOP_CLOSURE_EXCLUDE

        def _slam_state():
            kt = self.backend.keyframe_traj
            if kt is None or len(kt) == 0:
                all_xy = np.empty((0, 2))
                eligible = np.empty((0, 2))
            else:
                all_xy = kt[:, :2, 3]
                eligible = (
                    kt[:-LOOP_CLOSURE_EXCLUDE, :2, 3]
                    if len(kt) > LOOP_CLOSURE_EXCLUDE
                    else np.empty((0, 2))
                )
            return {
                "pose_idx": int(self.backend.pose_idx),
                "loop_closures": list(self.backend.loop_closures),
                "keyframe_xy_lc_eligible": eligible,
                "keyframe_xy_all": all_xy,
            }
        self.planner.set_slam_provider(_slam_state)

        goals_arr = np.round(np.asarray(goal_sequence, float), 2)
        print(
            f"[bold green]PerceptionAwarePlanner ready: {len(self.planner.waypoints)} waypoints, "
            f"start={np.round(np.asarray(start_xy, float), 2)} goals={goals_arr.tolist()} "
            f"alpha={self.planner.alpha} beta={self.planner.beta} "
            f"anchor_weight={self.planner.anchor_weight} "
            f"detour={'ON' if self.planner.detour_online_enabled else 'off'}"
            f"{' (' + self.planner.detour_target_mode + ')' if self.planner.detour_online_enabled else ''}"
        )

    def finalize(self):
        super().finalize()
        # Write anchor positions + detour events for offline diagnostics. The output dir matches
        # NavAgent.finalize's: output/NavAgent/<run_name>/ (nav_agent.get_entry_point hard-codes
        # "NavAgent" -- our subclass doesn't change that, so we use the same path here).
        try:
            out_dir = f"output/{_nav_get_entry_point()}/{self.run_name}"
            payload = {
                "start_xy": self.planner.start_xy.tolist(),
                "goal_sequence": self.planner._goal_sequence.tolist(),
                "anchors_xy": (
                    self.planner.perception_map.anchors_xy.tolist()
                    if getattr(self.planner.perception_map, "anchors_xy", None) is not None
                    else []
                ),
                "anchors_score": (
                    self.planner.perception_map.anchors_score.tolist()
                    if getattr(self.planner.perception_map, "anchors_score", None) is not None
                    else []
                ),
                "detour_attempts": int(self.planner.detour_attempts),
                "detour_successes": int(self.planner.detour_successes),
                "detour_events": self.planner.detour_events,
                "config_anchor_weight": float(self.planner.anchor_weight),
                "config_detour_online_enabled": bool(self.planner.detour_online_enabled),
                "config_detour_target_mode": str(self.planner.detour_target_mode),
                "config_detour_random_target": bool(self.planner.detour_random_target),
            }
            with open(f"{out_dir}/perception_planner_state.json", "w") as f:
                json.dump(payload, f, indent=2)
            print(
                f"[bold green]  wrote perception_planner_state.json "
                f"({len(payload['anchors_xy'])} anchors, "
                f"{payload['detour_attempts']} detours attempted, "
                f"{payload['detour_successes']} succeeded)"
            )
        except Exception as e:
            print(f"[bold red]  failed to write perception_planner_state.json: {e}")
