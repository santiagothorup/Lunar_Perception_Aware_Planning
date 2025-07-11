#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Data collection agent

"""

import numpy as np
import carla
import cv2 as cv
from pynput import keyboard
import signal
import os
from datetime import datetime
import random

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

from lac.planning.waypoint_planner import WaypointPlanner
from lac.control.steering import waypoint_steering
from lac.utils.data_logger import DataLogger
from lac.util import transform_to_numpy
import lac.params as params

# Attributes for teleop sensitivity and max speed
MAX_SPEED = 0.2
SPEED_INCREMENT = 0.05
TURN_RATE = 0.3

ARM_RAISE_WAIT_FRAMES = 80  # Number of frames to wait for the arms to raise

DISPLAY_IMAGES = True  # Set to False to disable image display
LOG_DATA = False  # Set to False to disable data logging

PRESET = os.environ.get("MISSIONS_SUBSET")
SEED = os.environ.get("SEED")


def get_entry_point():
    return "CarlaAgent"


class CarlaAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Set up a keyboard listener from pynput to capture the key commands for controlling the robot using the arrow keys."""
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        """ Add some attributes to store values for the target linear and angular velocity. """
        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """
        self.step = 0

        """ Planner """
        initial_pose = transform_to_numpy(self.get_initial_position())
        self.lander_pose = initial_pose @ transform_to_numpy(self.get_initial_lander_position())
        self.planner = WaypointPlanner(initial_pose, trajectory_type="spiral")

        # Camera config
        self.cameras = params.CAMERA_CONFIG_INIT
        self.cameras["FrontLeft"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": True,
        }
        self.cameras["FrontRight"] = {
            "active": True,
            "light": 1.0,
            "width": 1280,
            "height": 720,
            "semantic": True,
        }

        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()
        weather = self.world.get_weather()
        self.sun_azimuth_angle = weather.sun_azimuth_angle
        self.sun_altitude_angle = weather.sun_altitude_angle
        self.set_sun_direction(self.sun_azimuth_angle, self.sun_altitude_angle)

        blueprint_library = self.world.get_blueprint_library()
        print(f"Available blueprints: {[bp.id for bp in blueprint_library]}")
        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        controller_bp = blueprint_library.find("controller.ai.walker")
        spawn_point = carla.Transform(
            carla.Location(x=10.0, y=10.0, z=1.5),  # z=0.5 to prevent collision with ground
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        # walker = self.world.spawn_actor(controller_bp, spawn_point)

        vehicle_bp = random.choice(vehicle_blueprints)
        # # spawn_point = random.choice(self.world.get_map().get_spawn_points())
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # print(f"Spawned vehicle: {vehicle.type_id} at {spawn_point.location}")

        if LOG_DATA:
            agent_name = get_entry_point()
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.data_logger = DataLogger(self, agent_name, run_name, PRESET, SEED, self.cameras)

        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signal_received, frame):
        print("\nCtrl+C detected! Exiting mission")
        self.mission_complete()

    def use_fiducials(self):
        return False

    def set_sun_direction(self, azimuth: float, altitude: float):
        """Set the sun azimuth and altitude angles for the simulation."""
        weather = self.world.get_weather()
        weather.sun_azimuth_angle = azimuth
        weather.sun_altitude_angle = altitude
        self.world.set_weather(weather)

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light.
        """
        sensors = {}
        for cam, config in self.cameras.items():
            sensors[getattr(carla.SensorPosition, cam)] = {
                "camera_active": config["active"],
                "light_intensity": config["light"],
                "width": config["width"],
                "height": config["height"],
                "use_semantic": config["semantic"],
            }
        return sensors

    def image_available(self):
        return self.step % 2 == 0  # Image data is available every other step

    def initialize(self):
        # Move the arms out of the way
        self.set_front_arm_angle(np.deg2rad(75))
        self.set_back_arm_angle(np.deg2rad(75))

    def run_step(self, input_data):
        if self.step == 0:
            self.initialize()
        self.step += 1  # Starts at 0 at init, equal to 1 on the first run_step call
        print("\nStep: ", self.step)

        if self.image_available():
            if LOG_DATA:
                self.data_logger.log_images(self.step, input_data)
            if DISPLAY_IMAGES:
                FL_gray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
                cv.imshow("Front left", FL_gray)
                cv.waitKey(1)

        control = carla.VehicleVelocityControl(self.current_v, self.current_w)

        if LOG_DATA:
            self.data_logger.log_data(self.step, control)

        return control

    def finalize(self):
        print("Running finalize")
        if LOG_DATA:
            self.data_logger.save_log()

        """In the finalize method, we should clear up anything we've previously initialized that might be taking up memory or resources.
        In this case, we should close the OpenCV window."""
        if DISPLAY_IMAGES:
            cv.destroyAllWindows()

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += SPEED_INCREMENT
            self.current_v = np.clip(self.current_v, 0, MAX_SPEED)
        if key == keyboard.Key.down:
            self.current_v -= SPEED_INCREMENT
            self.current_v = np.clip(self.current_v, -MAX_SPEED, 0)
        if key == keyboard.Key.left:
            self.current_w = TURN_RATE
        if key == keyboard.Key.right:
            self.current_w = -TURN_RATE

        if hasattr(key, "char") and key.char is not None:
            if key.char == "l":
                print("A key pressed")
                self.sun_azimuth_angle += 1.0
            elif key.char == "j":
                self.sun_azimuth_angle -= 1.0
            elif key.char == "i":
                self.sun_altitude_angle += 1.0
            elif key.char == "k":
                self.sun_altitude_angle -= 1.0

            print(
                f"setting Sun azimuth angle: {self.sun_azimuth_angle}, Sun altitude angle: {self.sun_altitude_angle}"
            )
            self.set_sun_direction(self.sun_azimuth_angle, self.sun_altitude_angle)

    def on_release(self, key):
        """This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot."""

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()
