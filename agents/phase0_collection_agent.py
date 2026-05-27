#!/usr/bin/env python
"""Phase 0 data-collection agent (headless).

Drives the `phase0_transect` serpentine raster and logs FrontLeft + FrontRight grayscale images
plus ground-truth poses in exactly the format `scripts/phase0_validation.py` consumes
(`data_log.json` frames + `{cam}/{step:06}.png`). Unlike `data_collection_agent.py` this has no
teleop / display dependencies (no pynput, no cv.imshow), so it runs under `launch_headless.sh`
(Xvfb) without errors. Only the two cameras phase0 needs are enabled, grayscale-only, to keep the
render load minimal.

Run via `launch_phase0.sh` (sets TEAM_AGENT + MISSIONS_SUBSET=1 for preset 2). Output lands in
`LAC_SIM/output/Phase0CollectionAgent/<timestamp>/`.
"""

import os
import signal
from datetime import datetime

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.planning.waypoint_planner import WaypointPlanner
from lac.control.steering import waypoint_steering
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

        initial_pose = transform_to_numpy(self.get_initial_position())
        self.planner = WaypointPlanner(initial_pose, trajectory_type="phase0_transect")

        # Enable only the two cameras phase0_validation.py reads; grayscale, no semantic.
        self.cameras = {name: dict(cfg) for name, cfg in params.CAMERA_CONFIG_INIT.items()}
        for cam in ("FrontLeft", "FrontRight"):
            self.cameras[cam] = {
                "active": True, "light": 1.0, "width": 1280, "height": 720, "semantic": False,
            }

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

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1

        ground_truth_pose = transform_to_numpy(self.get_transform())
        waypoint, _ = self.planner.get_waypoint(self.step, ground_truth_pose, print_progress=True)
        if waypoint is None:  # raster complete -> end the mission
            self.mission_complete()
            return carla.VehicleVelocityControl(0.0, 0.0)

        if self.step < ARM_RAISE_WAIT_FRAMES:  # hold still until the arms are clear
            control = carla.VehicleVelocityControl(0.0, 0.0)
        else:
            steering = waypoint_steering(waypoint, ground_truth_pose)
            control = carla.VehicleVelocityControl(params.TARGET_SPEED, steering)

        # Log image-steps only (after the arm raise) so frames are 1:1 with saved images.
        if self.image_available() and self.step >= ARM_RAISE_WAIT_FRAMES:
            self.data_logger.log_images(self.step, input_data)
            self.data_logger.log_data(self.step, control)

        return control

    def finalize(self):
        print("Phase0CollectionAgent finalize - saving log")
        self.data_logger.save_log()
