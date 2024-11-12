import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

from typing import Dict, Union
from scipy.spatial.transform import Rotation as R
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.0,
}


class DroneEnv(MujocoEnv):

    def __init__(
        self,
        reward_coefficients=None,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        policy_freq=200,  # Default policy frequency (Hz)
        sim_steps_per_action=2,  # Default simulation steps between policy executions
        render_mode=None,
        visual_options={
            mujoco.mjtVisFlag.mjVIS_ACTUATOR: True,
            mujoco.mjtVisFlag.mjVIS_ACTIVATION: True,
        },
        **kwargs,
    ):

        # Path to your MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), "..", "scene.xml")

        self.DEFAULT_CAMERA_CONFIG = default_camera_config

        # Set parameters
        self.policy_freq = policy_freq
        self.sim_steps_per_action = sim_steps_per_action

        # Compute time per action
        self.time_per_action = 1.0 / self.policy_freq  # Time between policy executions

        # Set frame_skip to sim_steps_per_action
        frame_skip = self.sim_steps_per_action

        # Define action space: thrust inputs for the four motors
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.full(4, 0.11772, dtype=np.float32),
            dtype=np.float32,
        )

        # Update observation space to include position error
        obs_low = np.full(18, -np.inf, dtype=np.float32)
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.metadata["render_modes"] = [
            "human",
            "rgb_array",
            "depth_array",
        ]

        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            visual_options=visual_options,
            **kwargs,
        )

        # Set the simulation timestep
        self.model.opt.timestep = self.time_per_action / self.sim_steps_per_action

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Get the drone's body ID
        self.drone_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cf2"
        )

        # Simulation parameters
        self.workspace = {
            "low": np.array([-3.0, -3.0, 0.0]),
            "high": np.array([3.0, 3.0, 5.0]),
        }

        # Get the drone start position
        self.start_position = self.data.qpos[:3].copy()

        # Get the goal geometry ID
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )

        # Reward function coefficients
        if reward_coefficients is None:
            # Default coefficients
            self.reward_coefficients = {
                "distance_z": 0.5,
                "distance_xy": 0.2,
                "rotation_penalty": 1.0,
                "z_angular_velocity": 0.05,
                "angular_velocity": 0.1,
                "collision_penalty": 10.0,
                "terminate_collision": False,
                "out_of_bounds_penalty": 10.0,
                "alive_reward": 1.0,
                "linear_velocity": 0.1,
                "goal_bonus": 5.0,
                "distance": 0,
            }
        else:
            self.reward_coefficients = reward_coefficients

    def _get_obs(self):
        # Get observations
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:7].copy()  # Quaternion [w, x, y, z]
        linear_velocity = self.data.qvel[:3].copy()
        local_angular_velocity = self.data.qvel[3:6].copy()
        # local. See: https://github.com/google-deepmind/mujoco/issues/691

        # Convert quaternion to rotation matrix using scipy
        # MuJoCo quaternions are [w, x, y, z], scipy expects [x, y, z, w]
        orientation_q = np.array(
            [orientation[1], orientation[2], orientation[3], orientation[0]]
        )
        r = R.from_quat(orientation_q)
        # orientation_euler = r.as_euler('xyz', degrees=False)

        orientation_rot = r.as_matrix()

        # Compute position error in world coordinates
        position_error_world = self.target_position - position

        # Compute conjugate quaternion (inverse rotation)
        conj_quat = np.zeros(4)
        mujoco.mju_negQuat(conj_quat, orientation)

        # Rotate position error vector into drone's local frame
        position_error_local = np.zeros(3)
        mujoco.mju_rotVecQuat(position_error_local, position_error_world, conj_quat)

        # Rotate linear velocity into drone's local frame
        linear_velocity_local = np.zeros(3)
        mujoco.mju_rotVecQuat(linear_velocity_local, linear_velocity, conj_quat)

        # Combine all observations, including the position error in local frame
        obs = np.concatenate(
            [
                orientation_rot.flatten(),  # Orientation as rotation matrix. Flatten to 1D array with 9 elements
                linear_velocity_local,
                local_angular_velocity,
                position_error_local,  # Include position error in drone's local frame
            ]
        )

        return obs.astype(np.float32)

    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action and step simulation
        self.do_simulation(action, self.frame_skip)

        # Get observation
        obs = self._get_obs()

        # Extract state variables
        position = self.data.qpos[:3]
        orientation = self.data.qpos[3:7]  # Quaternion [w, x, y, z]
        angular_velocity = self.data.qvel[3:6]
        linear_velocity = self.data.qvel[:3]
        time = self.data.time

        # Check for collisions
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == self.drone_body_id or body2 == self.drone_body_id:
                collision = True
                break  # Exit the loop if a collision is detected

        # Check if out of bounds
        out_of_bounds = not np.all(self.workspace["low"] <= position) or not np.all(
            position <= self.workspace["high"]
        )

        # Compute reward
        reward, reward_components, additional_info = self.compute_reward(
            position,
            orientation,
            angular_velocity,
            linear_velocity,
            time,
            collision,
            out_of_bounds,
        )

        # Determine termination conditions
        terminated = False
        truncated = False
        if collision:
            terminated = self.reward_coefficients["terminate_collision"]
        if out_of_bounds:
            terminated = True  # Terminate the episode

        # Truncate episode if too long
        if self.data.time > 20:
            terminated = True
            truncated = True

        # Additional info
        info = {
            "position": position.copy(),
            "angular_velocity": angular_velocity.copy(),
            "reward_components": reward_components,
        }
        info.update(additional_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        position,
        orientation,
        angular_velocity,
        linear_velocity,
        time,
        collision,
        out_of_bounds,
    ):
        # Compute distance to target position
        distance = np.linalg.norm(position - self.target_position)

        # Compute rotation penalty, ignoring rotation around z-axis
        # Compute the body z-axis in world coordinates
        body_z_axis = np.array([0, 0, 1], dtype=np.float64)
        world_z_axis = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(world_z_axis, body_z_axis, orientation)

        # Normalize world_z_axis
        world_z_axis /= np.linalg.norm(world_z_axis)

        # Compute the angle between world_z_axis and global z-axis
        cos_theta = np.clip(world_z_axis[2], -1.0, 1.0)
        rotation_penalty = np.arccos(cos_theta)  # Penalty proportional to the angle

        # Compute angular velocity penalties
        z_angular_velocity = angular_velocity[2]
        angular_velocity_penalty = np.linalg.norm(angular_velocity)

        # Compute position errors
        distance_z = np.abs(position[2] - self.target_position[2])
        distance_xy = np.linalg.norm(position[:2] - self.target_position[:2])

        # Initialize reward components dictionary
        reward_components = {}

        # Initialize reward
        alive_reward = self.reward_coefficients["alive_reward"]  # Stay alive reward
        reward = alive_reward
        reward_components["alive_reward"] = alive_reward

        # If episode longer than 5s, focus on position tracking
        if time > 5:
            time_factor = min(time, 10)
            distance_penalty = time_factor * distance
            reward -= distance_penalty
            reward_components["distance_penalty"] = -distance_penalty

            z_angular_velocity_penalty = 1 * abs(z_angular_velocity)
            reward -= z_angular_velocity_penalty
            reward_components["z_angular_velocity_penalty"] = (
                -z_angular_velocity_penalty
            )

            linear_velocity_penalty = 2 * np.linalg.norm(linear_velocity)
            reward -= linear_velocity_penalty
            reward_components["linear_velocity_penalty"] = -linear_velocity_penalty
        else:
            reward_components["distance_penalty"] = 0
            reward_components["z_angular_velocity_penalty"] = 0
            reward_components["linear_velocity_penalty"] = 0

        # Subtract penalties and distances
        distance_z_penalty = self.reward_coefficients["distance_z"] * distance_z
        reward -= distance_z_penalty
        reward_components["distance_z_penalty"] = -distance_z_penalty

        distance_xy_penalty = self.reward_coefficients["distance_xy"] * distance_xy
        reward -= distance_xy_penalty
        reward_components["distance_xy_penalty"] = -distance_xy_penalty

        distance_penalty_coeff = self.reward_coefficients["distance"] * distance
        reward -= distance_penalty_coeff
        reward_components["distance_penalty_coeff"] = -distance_penalty_coeff

        rotation_penalty_value = (
            self.reward_coefficients["rotation_penalty"] * rotation_penalty
        )
        reward -= rotation_penalty_value
        reward_components["rotation_penalty"] = -rotation_penalty_value

        z_ang_vel_penalty_coeff = self.reward_coefficients["z_angular_velocity"] * abs(
            z_angular_velocity
        )
        reward -= z_ang_vel_penalty_coeff
        reward_components["z_angular_velocity_penalty_coeff"] = -z_ang_vel_penalty_coeff

        angular_vel_penalty_coeff = (
            self.reward_coefficients["angular_velocity"] * angular_velocity_penalty
        )
        reward -= angular_vel_penalty_coeff
        reward_components["angular_velocity_penalty"] = -angular_vel_penalty_coeff

        linear_vel_penalty_coeff = self.reward_coefficients[
            "linear_velocity"
        ] * np.linalg.norm(linear_velocity)
        reward -= linear_vel_penalty_coeff
        reward_components["linear_velocity_penalty_coeff"] = -linear_vel_penalty_coeff

        # Gaussian over distance for goal bonus
        if distance < 0.1:
            goal_bonus = self.reward_coefficients["goal_bonus"] * np.exp(
                -(distance**2) / 0.05**2
            )
            reward += goal_bonus
            reward_components["goal_bonus"] = goal_bonus
        else:
            reward_components["goal_bonus"] = 0

        # Initialize penalties
        if collision:
            collision_penalty = self.reward_coefficients["collision_penalty"]
            reward -= collision_penalty
            reward_components["collision_penalty"] = -collision_penalty
        else:
            reward_components["collision_penalty"] = 0

        if out_of_bounds:
            out_of_bounds_penalty = self.reward_coefficients["out_of_bounds_penalty"]
            reward -= out_of_bounds_penalty
            reward_components["out_of_bounds_penalty"] = -out_of_bounds_penalty
        else:
            reward_components["out_of_bounds_penalty"] = 0

        # Additional info
        additional_info = {
            "rotation_penalty": rotation_penalty,
            "distance_to_target_z": distance_z,
            "distance_to_target_xy": distance_xy,
            "distance_to_target": distance,
        }

        return reward, reward_components, additional_info

    def reset_model(self):
        # Randomize initial position around the target position
        position_std_dev = 0.5  # Standard deviation in meters
        random_position = self.np_random.normal(
            loc=self.target_position, scale=position_std_dev
        )
        # Clip the position to be within the workspace bounds
        random_position = np.clip(
            random_position, self.workspace["low"], self.workspace["high"]
        )
        self.data.qpos[:3] = random_position

        # Randomize initial orientation around upright orientation
        orientation_std_dev = np.deg2rad(30)  # Standard deviation of 30 degrees
        roll = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        pitch = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)  # Random yaw

        # Randomize velocity
        self.data.qvel[:3] = self.np_random.uniform(
            low=-0.4, high=0.4, size=3
        )  # Random linear velocity
        self.data.qvel[3:6] = self.np_random.uniform(
            low=-0.1, high=0.1, size=3
        )  # Random angular velocity

        # Convert Euler angles to quaternion
        euler = np.array([roll, pitch, yaw])
        q = np.zeros(4)
        seq = "xyz"  # Intrinsic rotations around x, y, z

        mujoco.mju_euler2Quat(q, euler, seq)
        mujoco.mju_normalize4(q)
        self.data.qpos[3:7] = q

        # Randomize initial actions in the action space
        initial_action = self.action_space.sample()
        self.data.ctrl[:] = initial_action

        # Update the goal marker position if you're displaying it
        # TODO: Update goal marker if needed

        mujoco.mj_forward(self.model, self.data)

        # Return initial observation
        obs = self._get_obs()
        return obs
