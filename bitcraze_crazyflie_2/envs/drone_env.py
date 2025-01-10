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
        policy_freq=200,  # Policy frequency in Hz
        sim_steps_per_action=2,  # Simulation steps between policy executions
        render_mode=None,
        visual_options=None,
        env_config={},
        target_move_prob=0.005,  # Probability of target moving when drone reaches it
        **kwargs,
    ):
        # Path to the MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), "mujoco", "scene_payload.xml")
        if not env_config.get("connect_payload", True):
            model_path = os.path.join(os.path.dirname(__file__), "mujoco", "scene.xml")
            

        self.DEFAULT_CAMERA_CONFIG = default_camera_config

        # Set parameters
        self.policy_freq = policy_freq
        self.sim_steps_per_action = sim_steps_per_action

        # Compute time per action
        self.time_per_action = 1.0 / self.policy_freq  # Time between policy executions

        # Set frame_skip to sim_steps_per_action
        frame_skip = self.sim_steps_per_action

        self.max_time = 20
        self.total_max_time = 100

        self.warmup_time = 0.3  # 1s warmup time

        # Define observation space
        obs_dim = 18  # Orientation matrix (9), linear velocity (3), angular velocity (3), position error (3), last action (4)
        obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.metadata["render_modes"] = ["human", "rgb_array", "depth_array"]

        # Set default visual options if none provided
        if visual_options is None:
            visual_options = {
                mujoco.mjtVisFlag.mjVIS_ACTUATOR: True,
                mujoco.mjtVisFlag.mjVIS_ACTIVATION: True,
            }

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



        self.max_thrust = 0.11772

        # Define action space: thrust inputs for the four motors
        self.action_space = spaces.Box(
            low=np.full(8, -1, dtype=np.float32),
            high=np.full(8, 1, dtype=np.float32),
            dtype=np.float32,
        )

        # Set the simulation timestep
        self.model.opt.timestep = self.time_per_action / self.sim_steps_per_action

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.5], dtype=np.float32)

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

        # Get payload body ID
        self.payload_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "payload"
        )

        # Reward function coefficients
        if reward_coefficients is None:
            # Default coefficients
            self.reward_coefficients = {
                "distance_z": 0.5,
                "distance_xy": 0.2,
                "rotation_penalty": 1.0,
                "z_angular_velocity": 0.1,
                "angular_velocity": 0.1,
                "collision_penalty": 10.0,
                "terminate_collision": False,
                "out_of_bounds_penalty": 10.0,
                "alive_reward": 1.0,
                "linear_velocity": 0.1,
                "goal_bonus": 5.0,
                "distance": 0,
                "velocity_towards_target": 4,
                "action_saturation": 100,
                "smooth_action": 100,
                "energy_penalty": 1,
            }
        else:
            self.reward_coefficients = reward_coefficients

        # Set the target move probability
        self.target_move_prob = target_move_prob

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

        # Extract IMU data
        imu_gyro_data = self.data.sensordata[:3]  # First 3 values (gyroscope)
        imu_acc_data = self.data.sensordata[3:6]  # Next 3 values (accelerometer)

        # Last action
        last_action = self.last_action if hasattr(self, "last_action") else np.zeros(4)

        # Combine all observations, including the position error in local frame
        obs = np.concatenate(
            [
                orientation_rot.flatten(),  # Orientation as rotation matrix. Flatten to 1D array with 9 elements
                linear_velocity_local,
                local_angular_velocity,
                position_error_local,  # Include position error in drone's local frame
                #last_action,
                # imu_gyro_data,
                # imu_acc_data,
            ]
        )

        obs = self.noise_observation(obs, noise_level=0.01)

        return obs.astype(np.float32)

    def noise_observation(self, obs, noise_level=0.01):

        obs += np.random.normal(loc=0, scale=noise_level, size=obs.shape)
        return obs

    def step(self, action):

        # Split the action
        action_mean = action[:4]
        log_std_unscaled = action[4:]

        # Map [-1,1] -> [-3,1] for log(std)
        action_log_std = -3.0 + (log_std_unscaled + 1.0) * 2.0
        action_std = np.exp(action_log_std)

        # Sample from Gaussian
        action_sample = np.random.normal(loc=action_mean, scale=action_std)

        # Scale action to [0, self.max_thrust]
        action_sample = 0.5 * (action_sample + 1.0) * self.max_thrust
        action_sample = np.clip(action_sample, 0, self.max_thrust)

        self.do_simulation(action_sample, self.frame_skip)

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
        reward, reward_components, additional_info = self.calc_reward(
            position,
            orientation,
            angular_velocity,
            linear_velocity,
            time,
            collision,
            out_of_bounds,
            action_sample,
        )

        # Determine termination conditions
        terminated = False
        truncated = False
        if collision:
            terminated = self.reward_coefficients["terminate_collision"]
        if out_of_bounds:
            terminated = True  # Terminate the episode

        # Truncate episode if too long
        if (
            self.data.time - self.warmup_time > self.max_time
            or self.data.time - self.warmup_time > self.total_max_time
        ):
            terminated = True
            truncated = True

        # Additional info
        info = {
            "position": position.copy(),
            "angular_velocity": angular_velocity.copy(),
            "reward_components": reward_components,
            "action": action_sample,
        }
        info.update(additional_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def calc_reward(
        self,
        position,
        orientation,
        angular_velocity,
        linear_velocity,
        time,
        collision,
        out_of_bounds,
        action,
    ):
        # Compute distance to target position
        position_error = self.target_position - position
        distance = np.linalg.norm(position_error)

        # Compute rotation penalty

        body_z_axis = np.array([0, 0, 1], dtype=np.float64)
        world_z_axis = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(world_z_axis, body_z_axis, orientation)

        # Compute the norm of world_z_axis
        norm_world_z_axis = np.linalg.norm(world_z_axis)

        # Prevent division by zero
        if norm_world_z_axis < 1e-8:
            # Handle the zero vector case
            # Set world_z_axis to the global z-axis
            world_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            # Normalize world_z_axis
            world_z_axis /= norm_world_z_axis

        cos_theta = np.clip(world_z_axis[2], -1.0, 1.0)
        rotation_penalty = np.arccos(cos_theta)

        # Compute angular velocity penalties
        z_angular_velocity = angular_velocity[2]
        angular_velocity_penalty = np.linalg.norm(angular_velocity)

        # Compute position errors
        distance_z = np.abs(position[2] - self.target_position[2])
        distance_xy = np.linalg.norm(position[:2] - self.target_position[:2])

        # Compute action saturation penalty
        action_saturation = (
            np.sum(
                np.max(np.exp(-0.5 * ((action + 0.001) / 0.001) ** 2)
                + np.exp(-0.5 * ((action - self.max_thrust - 0.001) / 0.001) ** 2) - 0.01, 0)
            )
            / 4
        )

        # Initialize reward components
        reward_components = {}

        rc = self.reward_coefficients
        reward = rc["alive_reward"]
        reward_components["alive_reward"] = reward

        # Subtract penalties and distances
        distance_z_penalty = rc["distance_z"] * distance_z
        reward -= distance_z_penalty
        reward_components["distance_z_penalty"] = -distance_z_penalty

        distance_xy_penalty = rc["distance_xy"] * distance_xy
        reward -= distance_xy_penalty
        reward_components["distance_xy_penalty"] = -distance_xy_penalty

        distance_penalty = rc["distance"] * distance
        reward -= distance_penalty
        reward_components["distance_penalty"] = -distance_penalty

        rotation_penalty_value = rc["rotation_penalty"] * rotation_penalty
        reward -= rotation_penalty_value
        reward_components["rotation_penalty"] = -rotation_penalty_value

        z_ang_vel_penalty = rc["z_angular_velocity"] * abs(z_angular_velocity)
        reward -= z_ang_vel_penalty
        reward_components["z_angular_velocity_penalty"] = -z_ang_vel_penalty

        angular_vel_penalty = rc["angular_velocity"] * angular_velocity_penalty
        reward -= angular_vel_penalty
        reward_components["angular_velocity_penalty"] = -angular_vel_penalty

        action_saturation_penalty = rc["action_saturation"] * action_saturation
        reward -= action_saturation_penalty
        reward_components["action_saturation_penalty"] = -action_saturation_penalty

        action_energy_penalty = rc["energy_penalty"] * np.sum(action)**2
        reward -= action_energy_penalty
        reward_components["action_energy_penalty"] = -action_energy_penalty

        # Smooth action penalty
        if hasattr(self, "last_action"):
            action_difference_penalty = rc["smooth_action"] * np.sum(
                (action - self.last_action)
            )
            reward -= action_difference_penalty
            reward_components["action_difference_penalty"] = -action_difference_penalty**2
        self.last_action = action

        # Move towards target
        # Compute the unit vector towards the target
        if distance > 1e-8:
            desired_direction = position_error / distance
        else:
            desired_direction = np.zeros_like(position_error)

        # Compute the velocity towards the target
        velocity_towards_target = np.dot(linear_velocity, desired_direction)

        reward += rc["velocity_towards_target"] * velocity_towards_target
        reward_components["velocity_towards_target"] = (
            rc["velocity_towards_target"] * velocity_towards_target
        )

        # Linear velocity penalty
        linear_vel_penalty = rc["linear_velocity"] * (
            np.linalg.norm(velocity_towards_target) - distance
        )
        reward -= linear_vel_penalty
        reward_components["linear_velocity_penalty"] = -linear_vel_penalty

        # Goal bonus
        if distance < 0.1:
            goal_bonus = (
                0.5 * rc["goal_bonus"] * np.exp(-(distance**2) / 0.008**2)
            )  # exact peek at position
            goal_bonus += 0.5 * rc["goal_bonus"] * np.exp(-(distance**2) / 0.04**2)

            # Move the target if good tracking
            if distance < 0.03:
                if self.np_random.uniform() < self.target_move_prob:
                    # Move the target to a new random position
                    self.target_position = self.np_random.uniform(
                        low=self.workspace["low"] + 0.1,
                        high=self.workspace["high"] - 0.1,
                    )
                    # Update the goal marker's position if applicable
                    self.model.geom_pos[self.goal_geom_id] = self.target_position

                    self.max_time += 10  # add more time to reach the target
                    goal_bonus += (
                        1000 * rc["goal_bonus"]
                    )  # add more bonus for reaching the target and to prevent policy avoiding it

            reward += goal_bonus
            reward_components["goal_bonus"] = goal_bonus
        else:
            reward_components["goal_bonus"] = 0

        # Collision penalty
        if collision:
            collision_penalty = rc["collision_penalty"]
            reward -= collision_penalty
            reward_components["collision_penalty"] = -collision_penalty
        else:
            reward_components["collision_penalty"] = 0

        # Out of bounds penalty
        if out_of_bounds:
            out_of_bounds_penalty = rc["out_of_bounds_penalty"]
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
        position_std_dev = 0.2  # Standard deviation in meters
        random_position = self.np_random.normal(
            loc=self.target_position- np.array([0.4,0,0]), scale=position_std_dev
        )
        # Clip the position to be within the workspace bounds
        random_position = np.clip(
            random_position, self.workspace["low"], self.workspace["high"]
        )
        self.data.qpos[:3] = random_position

        # Randomize initial orientation around upright orientation
        orientation_std_dev = np.deg2rad(20)  # Standard deviation of 30 degrees
        roll = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        pitch = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)  # Random yaw

        # Convert Euler angles to quaternion
        euler = np.array([roll, pitch, yaw])
        q = np.zeros(4)
        seq = "xyz"  # Intrinsic rotations around x, y, z

        mujoco.mju_euler2Quat(q, euler, seq)
        mujoco.mju_normalize4(q)
        self.data.qpos[3:7] = q

        # set payload position
        payload_joint_id = self.model.body_jntadr[self.payload_body_id]
        payload_qpos_index = self.model.jnt_qposadr[payload_joint_id]
        self.data.qpos[payload_qpos_index : payload_qpos_index + 3] = (
            random_position + np.array([0, 0, -0.15])
        )

        # warmup sim to stabilize rope
        while self.data.time < self.warmup_time:
            self.do_simulation(np.zeros(4), 10)
            # reset qpos
            self.data.qpos[:3] = random_position
            self.data.qpos[3:7] = q

        self.warmup_time = self.data.time

        # Randomize velocity
        self.data.qvel[:3] = self.np_random.uniform(
            low=-0.4, high=0.4, size=3
        )  # Random linear velocity
        self.data.qvel[3:6] = self.np_random.uniform(
            low=-0.1, high=0.1, size=3
        )  # Random angular velocity

        # Randomize initial actions in the action space
        initial_action = self.action_space.sample()
        self.data.ctrl[:] = initial_action[:4]

        # Update the goal marker position
        self.model.geom_pos[self.goal_geom_id] = self.target_position

        mujoco.mj_forward(self.model, self.data)

        # Return initial observation
        obs = self._get_obs()
        return obs
