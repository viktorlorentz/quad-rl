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
        policy_freq=200,             # Default policy frequency (Hz)
        sim_steps_per_action=2,       # Default simulation steps between policy executions
        render_mode=None,
        **kwargs,
    ):

        # Path to your MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'scene.xml')

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
            dtype=np.float32
        )

        # Update observation space to include position error
        obs_low = np.full(18, -np.inf, dtype=np.float32)
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.metadata["render_modes"] = [
            "human",
            "rgb_array",
            "depth_array",
        ]

        # Initialize MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip = frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs,
        )

        # Set the simulation timestep
        self.model.opt.timestep = self.time_per_action / self.sim_steps_per_action

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Get the ID of the goal site
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'goal_site')

        # Get the drone's body ID
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cf2')

        # Simulation parameters
        self.workspace = {
            'low': np.array([-3.0, -3.0, 0.0]),
            'high': np.array([3.0, 3.0, 5.0])
        }

        # Get the drone start position
        self.start_position = self.data.qpos[:3].copy()

        # Get the goal geometry ID
        self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker')

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
                "out_of_bounds_penalty": 10.0,
                "alive_reward": 1.0,
            }
        else:
            self.reward_coefficients = reward_coefficients

    def _get_obs(self):
        # Get observations
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:7].copy()  # Quaternion [w, x, y, z]
        linear_velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()

        # Convert quaternion to Euler angles using scipy
        # MuJoCo quaternions are [w, x, y, z], scipy expects [x, y, z, w]
        orientation_q = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(orientation_q)
        #orientation_euler = r.as_euler('xyz', degrees=False)

        orientation_rot = r.as_matrix()


        # Compute position error in world coordinates
        position_error_world = self.target_position - position

        # Compute conjugate quaternion (inverse rotation)
        conj_quat = np.zeros(4)
        mujoco.mju_negQuat(conj_quat, orientation)

        # Rotate position error vector into drone's local frame
        position_error_local = np.zeros(3)
        mujoco.mju_rotVecQuat(position_error_local, position_error_world, conj_quat)

        # Combine all observations, including the position error in local frame
        obs = np.concatenate([
            orientation_rot.flatten(),  # Orientation as rotation matrix. Flatten to 1D array wiht 9 elements
            linear_velocity,
            angular_velocity,
            position_error_local  # Include position error in drone's local frame
        ])

        return obs.astype(np.float32)

    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action and step simulation
        self.do_simulation(action, self.frame_skip)

        # Get observation
        obs = self._get_obs()

        # Extract position and orientation
        position = self.data.qpos[:3]
        orientation = self.data.qpos[3:7]  # Quaternion [w, x, y, z]

        # Compute distance to target position
        # distance = np.linalg.norm(position - self.target_position)

        # Compute rotation penalty, ignoring rotation around z-axis
        # Compute the body z-axis in world coordinates
        body_z_axis = np.array([0, 0, 1], dtype=np.float64)
        world_z_axis = np.zeros(3, dtype=np.float64)

        mujoco.mju_rotVecQuat(world_z_axis, body_z_axis, orientation)

        # Normalize world_z_axis
        world_z_axis /= np.linalg.norm(world_z_axis)

        # Compute the angle between world_z_axis and global z-axis
        cos_theta = np.clip(world_z_axis[2], -1.0, 1.0)
        angle = np.arccos(cos_theta)
        rotation_penalty = angle  # Penalty proportional to the angle

        # orientation_euler = obs[:3]

        # # penalize roll and pitch
        # rotation_penalty = np.abs(orientation_euler[0]) + np.abs(orientation_euler[1])

        # Compute angular velocity penalty
        angular_velocity = self.data.qvel[3:6]
        z_angular_velocity = angular_velocity[2]
        angular_velocity_penalty = np.linalg.norm(angular_velocity)

        # Compute position error
        position_error = obs[9:12]
        distance_z = np.abs(position_error[2])
        distance_xy = np.linalg.norm(position[:2] - self.target_position[:2])

        # Initialize reward
        reward = self.reward_coefficients["alive_reward"]  # Stay alive reward

        # Subtract penalties and distances
        reward -= self.reward_coefficients["distance_z"] * distance_z
        reward -= self.reward_coefficients["distance_xy"] * distance_xy
        reward -= self.reward_coefficients["rotation_penalty"] * rotation_penalty
        reward -= self.reward_coefficients["z_angular_velocity"] * abs(z_angular_velocity)
        reward -= self.reward_coefficients["angular_velocity"] * angular_velocity_penalty

        # Check for collisions
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == self.drone_body_id or body2 == self.drone_body_id:
                collision = True
                break  # Exit the loop if a collision is detected

        terminated = False
        truncated = False

        if collision:
            terminated = True
            reward -= self.reward_coefficients["collision_penalty"]

        # Check if out of bounds
        if not np.all(self.workspace['low'] <= position) or not np.all(position <= self.workspace['high']):
            terminated = True  # Uncomment if you want to terminate
            reward -= self.reward_coefficients["out_of_bounds_penalty"]

        # Additional info
        info = {
            'position': position.copy(),
            'distance_to_target': distance_z,
            'rotation_penalty': rotation_penalty,
            'angular_velocity': angular_velocity.copy()
        }

        if self.render_mode == 'human':
            self.render()

        # Truncate episode if too long
        if self.data.time > 20:
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # Randomize initial position around the target position
        position_std_dev = 0.1  # Standard deviation of 0.1 meters
        random_position = self.np_random.normal(loc=self.target_position, scale=position_std_dev)
        # Clip the position to be within the workspace bounds
        random_position = np.clip(random_position, self.workspace['low'], self.workspace['high'])
        self.data.qpos[:3] = random_position

        # Randomize initial orientation around upright orientation
        orientation_std_dev = np.deg2rad(20)  # Standard deviation of 10 degrees
        roll = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        pitch = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)  # Random yaw

        #ramdomize velocity
        self.data.qvel[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.data.qvel[3:6] = self.np_random.uniform(low=-0.1, high=0.1, size=3)


        # Convert Euler angles to quaternion
        euler = np.array([roll, pitch, yaw])
        q = np.zeros(4)
        seq = 'xyz'  # Intrinsic rotations around x, y, z

        mujoco.mju_euler2Quat(q, euler, seq)
        mujoco.mju_normalize4(q)
        self.data.qpos[3:7] = q

        # Reset velocities to zero
        self.data.qvel[:] = 0

        # Randomize initial actions in the action space
        initial_action = self.action_space.sample()
        self.data.ctrl[:] = initial_action

        # Update the goal marker position if you're displaying it
        if self.goal_site_id is not None:
            self.model.site_pos[self.goal_site_id] = self.target_position
        mujoco.mj_forward(self.model, self.data)

        # Return initial observation
        obs = self._get_obs()
        return obs
