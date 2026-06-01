#!/usr/bin/env python
"""Phase 0 data-collection agent (headless, obstacle-avoiding).

Drives the `phase0_transect` serpentine raster and logs FrontLeft + FrontRight grayscale images
plus ground-truth poses in exactly the format `scripts/phase0_validation.py` consumes
(`data_log.json` frames + `{cam}/{step:06}.png`).

Navigation is by GROUND-TRUTH pose (so look-ahead DEM lookups in the validation are exact), but
steering goes through the same ArcPlanner obstacle-avoidance system `nav_agent` uses: each image
step runs the perception Frontend to get rock detections and feeds them (plus the lander bbox) to
`ArcPlanner.plan_arc`, so the rover routes AROUND rocks instead of driving into them. A naive
`waypoint_steering` collector (the previous version) flipped on a large rock on preset 7 and
recorded ~5k black sky frames; this avoids that. A stuck/backup maneuver is included so no single
rock can pin the rover. The SLAM backend is intentionally not run -- only `rock_data` is needed.

Run via `launch_phase0.sh` (sets TEAM_AGENT + MISSIONS_SUBSET). Output lands in
`LAC_SIM/output/Phase0CollectionAgent/<timestamp>/`.
"""

import os
import signal
from collections import deque
from datetime import datetime

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.planning.waypoint_planner import WaypointPlanner
from lac.planning.arc_planner import ArcPlanner
from lac.slam.semantic_feature_tracker import SemanticFeatureTracker
from lac.slam.frontend import Frontend
from lac.utils.data_logger import DataLogger
from lac.util import transform_to_numpy
import lac.params as params

ARM_RAISE_WAIT_FRAMES = 80  # hold still until the arms swing clear of the cameras

PRESET = os.environ.get("MISSIONS_SUBSET")  # logged as metadata only
SEED = os.environ.get("SEED")


def get_entry_point():
    return "Phase0CollectionAgent"


class Phase0CollectionAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.step = 0
        self.current_v = 0.0
        self.current_w = 0.0
        self.current_velocity = np.zeros(3)
        self.backup_counter = 0
        self.stuck_counter = 0
        self.stuck_timer = 0
        self.imu_measurements = deque(maxlen=2)

        self.initial_pose = transform_to_numpy(self.get_initial_position())
        # "phase0_transect" (default) for full data collection; "phase0_probe" for a quick run that
        # only triggers the leaderboard's ground-truth DEM save (preset terrain-richness scan).
        trajectory = os.environ.get("PHASE0_TRAJECTORY", "phase0_transect")
        self.planner = WaypointPlanner(self.initial_pose, trajectory_type=trajectory)

        # Enable only the two cameras phase0_validation.py reads + the stereo pair the Frontend needs
        # for rock detection; grayscale, no semantic outputs.
        self.cameras = {name: dict(cfg) for name, cfg in params.CAMERA_CONFIG_INIT.items()}
        for cam in ("FrontLeft", "FrontRight"):
            self.cameras[cam] = {
                "active": True, "light": 1.0, "width": 1280, "height": 720, "semantic": False,
            }
        self.active_cameras = [cam for cam, cfg in self.cameras.items() if cfg["active"]]

        # Perception Frontend (for rock_data) + ArcPlanner (rock + lander avoidance). No backend:
        # we navigate by ground truth, so SLAM pose estimation is unnecessary.
        feature_tracker = SemanticFeatureTracker(self.cameras)
        self.frontend = Frontend(feature_tracker, initial_pose=self.initial_pose)
        self.arc_planner = ArcPlanner()

        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.data_logger = DataLogger(self, get_entry_point(), run_name, PRESET, SEED, self.cameras)

        # Save the log on Ctrl-C (signal is headless-safe; no pynput).
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\nInterrupt received - ending mission.")
        self.mission_complete()

    def use_fiducials(self):
        return False

    def sensors(self):
        return {
            getattr(carla.SensorPosition, cam): {
                "camera_active": cfg["active"],
                "light_intensity": cfg["light"],
                "width": cfg["width"],
                "height": cfg["height"],
                "use_semantic": cfg["semantic"],
            }
            for cam, cfg in self.cameras.items()
        }

    def image_available(self):
        return self.step % 2 == 0  # image data is produced every other step

    def initialize(self):
        # Swing both arms out of the cameras' field of view.
        self.set_front_arm_angle(np.deg2rad(75))
        self.set_back_arm_angle(np.deg2rad(75))

    def run_backup_maneuver(self):
        """Back up then rotate to escape a stuck spot (mirrors nav_agent)."""
        self.backup_counter += 1
        BACKUP_TIME, ROTATE_TIME = 5.0, 2.0  # [s]
        if self.backup_counter <= params.FRAME_RATE * BACKUP_TIME:
            return carla.VehicleVelocityControl(-0.2, 0.0)
        elif self.backup_counter <= params.FRAME_RATE * (BACKUP_TIME + ROTATE_TIME):
            return carla.VehicleVelocityControl(0.0, np.pi / 4)
        self.backup_counter = 0
        return carla.VehicleVelocityControl(self.current_v, self.current_w)

    def check_stuck(self):
        """True if the rover has been near-stationary for >50% of a 2-3 s window (GT speed)."""
        if self.step < ARM_RAISE_WAIT_FRAMES + 10:
            return False
        is_stuck = self.get_linear_speed() < 0.25 * params.TARGET_SPEED
        if is_stuck and self.stuck_timer == 0:
            self.stuck_counter += 1
            self.stuck_timer += 1
        elif is_stuck and self.stuck_timer > 0:
            self.stuck_counter += 1
        if params.FRAME_RATE * 2 < self.stuck_timer < params.FRAME_RATE * 3:
            if (self.stuck_counter / self.stuck_timer) > 0.5:
                self.stuck_counter = self.stuck_timer = 0
                return True
        elif self.stuck_timer >= params.FRAME_RATE * 3:
            if (self.stuck_counter / self.stuck_timer) < 0.5:
                self.stuck_counter = self.stuck_timer = 0
        return False

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1
        if self.stuck_timer > 0:
            self.stuck_timer += 1

        ground_truth_pose = transform_to_numpy(self.get_transform())
        waypoint, _ = self.planner.get_waypoint(self.step, ground_truth_pose, print_progress=True)
        if waypoint is None:  # raster complete -> end the mission
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        self.imu_measurements.append(self.get_imu_data())
        control = (0.0, 0.0)

        if self.image_available():
            images = {cam: input_data["Grayscale"][getattr(carla.SensorPosition, cam)]
                      for cam in self.active_cameras}
            if self.step >= ARM_RAISE_WAIT_FRAMES:
                if self.step == ARM_RAISE_WAIT_FRAMES:
                    self.frontend.initialize(images)
                else:
                    images["step"] = self.step
                    images["imu_measurements"] = self.imu_measurements
                    images["prev_pose"] = ground_truth_pose
                    # Frontend gives rock detections; ArcPlanner picks a rock/lander-free arc toward
                    # the waypoint. (VO odometry from process_frame is unused -- we steer by GT pose.)
                    data = self.frontend.process_frame(images)
                    control, _, _ = self.arc_planner.plan_arc(
                        waypoint, ground_truth_pose, data["rock_data"]
                    )
                    if control is not None:
                        self.current_v, self.current_w = control

        # Control selection: hold during arm raise; back up if stuck or no safe arc; else drive.
        if self.step < ARM_RAISE_WAIT_FRAMES:
            carla_control = carla.VehicleVelocityControl(0.0, 0.0)
        elif self.backup_counter > 0 or self.check_stuck():
            carla_control = self.run_backup_maneuver()
        elif control is None:
            carla_control = self.run_backup_maneuver()
        else:
            carla_control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        # Log image-steps only (after the arm raise) so frames are 1:1 with saved images.
        if self.image_available() and self.step >= ARM_RAISE_WAIT_FRAMES:
            self.data_logger.log_images(self.step, input_data)
            self.data_logger.log_data(self.step, carla_control)

        return carla_control

    def finalize(self):
        print("Phase0CollectionAgent finalize - saving log")
        self.data_logger.save_log()
