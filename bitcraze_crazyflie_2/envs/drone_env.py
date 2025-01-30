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

MAX_THRUST = 0.11772

#src: https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/gym_art/quadrotor_multi/quad_utils.py
class OUNoise:
    def __init__(self, size=4, mu=0.0, theta=0.15, sigma=0.2 ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DroneEnv(MujocoEnv):
    def __init__(
        self,
        reward_coefficients=None,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        policy_freq=200,  # Policy frequency in Hz
        sim_steps_per_action=1,  # Simulation steps between policy executions
        render_mode=None,
        visual_options=None,
        env_config={},
        target_move_prob=0.01,  # Probability of target moving when drone reaches it
        **kwargs,
    ):
        # Path to the MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), "mujoco", "scene_payload.xml")

        self.payload = env_config.get("connect_payload", True)

        if not self.payload:
            model_path = os.path.join(os.path.dirname(__file__), "mujoco", "scene.xml")
        
        self.curriculum = env_config.get("curriculum", False)
        if  self.curriculum:
        
            self.randomness_max = env_config.get("randomness", 1.0)

            self.randomness = 0.0
        else:
            self.randomness = env_config.get("randomness", 0.0)
          

        self.average_episode_length = 0

       

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

        self.warmup_time = 0.0  # 1s warmup time


        # Define observation space
        obs_dim = 22  # Orientation matrix (9), linear velocity (3), angular velocity (3), position error (3), last action (4)
        
        # add paylod pos if available
        if self.payload:
            obs_dim += 6 # relative payload pos (3), payload vel (3)
        
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
            low=np.full(4, -1, dtype=np.float32),
            high=np.full(4, 1, dtype=np.float32),
            dtype=np.float32,
        )

        # Set the simulation timestep
        self.model.opt.timestep = self.time_per_action / self.sim_steps_per_action

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Set the target position for hovering
        self.target_mode = env_config.get("target_mode", "quad") # quad or payload
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
                "payload_velocity": 0.2,
                "above_payload": 1
            }
        else:
            self.reward_coefficients = reward_coefficients

        # Set the target move probability
        self.target_move_prob = target_move_prob

        self.thrust_noise_ratio = 0.05
        self.ou_noise = OUNoise(size=self.action_space.shape, sigma=self.thrust_noise_ratio)  # OU noise for motor signals
        self.motor_tau_up = 0.2
        self.motor_tau_down = 0.3
        self.current_thrust = np.zeros(4)

    def R_from_quat(self, q):
        # Convert mujoco quaternion to rotation matrix explicitly
        # MuJoCo quaternions are [w, x, y, z]
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        r = np.array(
            [
                [
                    1 - 2 * (y ** 2 + z ** 2),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x ** 2 + z ** 2),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x ** 2 + y ** 2),
                ],
            ]
        )
        return r
    
    def to_frame(self, orientation_q, vec):
        # Convert a vector from the world frame to the body frame
        # quat: quaternion of the body frame
        # vec: vector in the world frame
        # with explicit math so its implemetable in C

        q = orientation_q
        R = self.R_from_quat(q)
        
        # inverse rotation
        R_inv = np.array(
            [
                [R[0, 0], R[1, 0], R[2, 0]],
                [R[0, 1], R[1, 1], R[2, 1]],
                [R[0, 2], R[1, 2], R[2, 2]],
            ]
        )
        vec = np.array(
            [
                R_inv[0, 0] * vec[0] + R_inv[0, 1] * vec[1] + R_inv[0, 2] * vec[2],
                R_inv[1, 0] * vec[0] + R_inv[1, 1] * vec[1] + R_inv[1, 2] * vec[2],
                R_inv[2, 0] * vec[0] + R_inv[2, 1] * vec[1] + R_inv[2, 2] * vec[2],
            ]
            )
        return vec



    def _get_obs(self):
        # Get observations
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:7].copy()  # Quaternion [w, x, y, z]
        linear_velocity = self.data.qvel[:3].copy()
        local_angular_velocity = self.data.qvel[3:6].copy()
        # local. See: https://github.com/google-deepmind/mujoco/issues/691

        # Convert quaternion to rotation matrix
        orientation_rot = self.R_from_quat(orientation)
        # Orientation as rotation matrix. Flatten to 1D array with 9 elements. 
        # Numpy flattesn row-wise so its 
        # [r11, r12, r13, r21, r22, r23, r31, r32, r33]
        orientation_rot = orientation_rot.flatten()
        # Compute position error in world coordinates
        if self.target_mode == "payload" and self.payload:
            payload_joint_id = self.model.body_jntadr[self.payload_body_id]
            payload_pos = self.data.qpos[payload_joint_id : payload_joint_id + 3]
            position_error_world = self.target_position - payload_pos
        else:
            position_error_world = self.target_position - position


        # Rotate position error vector into drone's local frame
        position_error_local = self.to_frame(orientation, position_error_world)
        # Rotate linear velocity into drone's local frame
        linear_velocity_local = self.to_frame(orientation, linear_velocity)
        # Extract IMU data
        # imu_gyro_data = self.data.sensordata[:3]  # First 3 values (gyroscope)
        # imu_acc_data = self.data.sensordata[3:6]  # Next 3 values (accelerometer)

        # Last action
        last_action = self.last_action if hasattr(self, "last_action") else np.zeros(4)

        obs = [     
            orientation_rot,
            linear_velocity_local,
            local_angular_velocity,
            position_error_local,  # Include position error in drone's local frame
            last_action,
            ]
        
        #payload pos
        if self.payload:
            payload_joint_id = self.model.body_jntadr[self.payload_body_id]
            payload_pos = self.data.qpos[payload_joint_id : payload_joint_id + 3]
            relative_payload_pos = payload_pos - position
            
            relative_payload_pos_local = self.to_frame(orientation, relative_payload_pos)
            obs.append(relative_payload_pos_local)
            
            payload_vel = self.data.qvel[payload_joint_id : payload_joint_id + 3]
            payload_vel_local = self.to_frame(orientation, payload_vel)
            obs.append(payload_vel_local)
           
           

        
        # Combine all observations, including the position error in local frame
        obs = np.concatenate(obs)

        obs = self.noise_observation(obs, noise_level=0.02)

        return obs.astype(np.float32)

    def noise_observation(self, obs, noise_level=0.02):

        obs += np.random.normal(loc=0, scale=noise_level, size=obs.shape)
        return obs
    
    

    def step(self, action):
        if not hasattr(self, "thrust_rot_damp"):
            self.thrust_rot_damp = np.zeros(4)
        if not hasattr(self, "thrust_cmds_damp"):
            self.thrust_cmds_damp = np.zeros(4)

        # Convert action from [-1,1] to [0,1]
        thrust_cmds = 0.5 * (action + 1.0)
        thrust_cmds = np.clip(thrust_cmds, 0.0, 1.0)

        action_scaled = thrust_cmds * self.max_thrust

        # Motor tau
        motor_tau = self.motor_tau_up * np.ones(4)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        # Convert to sqrt scale
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
       

        # Add noise
        thr_noise = self.ou_noise.noise()
        thrust_noise = thrust_cmds * thr_noise
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        self.thrust_cmds_damp = self.thrust_rot_damp ** 2
        
        # Scale to actual thrust
        self.current_thrust = self.max_thrust * self.thrust_cmds_damp 

        # Apply motor offset
        self.current_thrust *= self.motor_offset

        # Store last action
        self.last_action = (self.data.ctrl[:4].copy() / self.max_thrust) * 2.0 - 1.0
        
        #reset position and orientation to 0
        # self.data.qpos[:3] = [0, 0 ,1]
        # self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        
        # Run simulation
        self.do_simulation(self.current_thrust, self.frame_skip)

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
            action_scaled,
            last_action= 0.5 * (obs[-4:] + 1.0) * self.max_thrust,
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
            "action": action_scaled,
        }
        info.update(additional_info)

        # Update average episode length and env randomness
        if terminated and self.curriculum:
            self.average_episode_length = (
                self.average_episode_length * 0.95 + (self.data.time - self.warmup_time) * 0.05
            )
            if self.average_episode_length >  0.9 * self.max_time and self.randomness < self.randomness_max:
                self.randomness += 0.01
            
            if self.average_episode_length < 0.45 * self.max_time and self.randomness > 0.0:
                self.randomness -= 0.01
            info["env_randomness"] = self.randomness
            info["average_episode_length"] = self.average_episode_length


        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def angle_between(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        return np.arccos(cos_theta)

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
        last_action,
    ):
        # Compute distance to target position
        if self.target_mode == "payload" and self.payload:
            payload_joint_id = self.model.body_jntadr[self.payload_body_id]
            payload_pos = self.data.qpos[payload_joint_id : payload_joint_id + 3]
            position_error = self.target_position - payload_pos
        else:
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
        action_saturation = np.mean(
                np.max(np.exp(-0.5 * ((action + 0.001) / 0.001) ** 2)
                + np.exp(-0.5 * ((action - self.max_thrust - 0.001) / 0.001) ** 2) - 0.01, 0)
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

        distance_penalty = rc["distance"] * distance**2
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

        action_energy_penalty = rc["energy_penalty"] * np.mean(action/self.max_thrust)**2
        reward -= action_energy_penalty
        reward_components["action_energy_penalty"] = -action_energy_penalty

        # Smooth action penalty
        if hasattr(self, "last_action"):
            action_difference_penalty = rc["smooth_action"] * np.mean(
                np.abs(action - last_action)/self.max_thrust
            )**2
            reward -= action_difference_penalty
            reward_components["action_difference_penalty"] = -action_difference_penalty

        # Move towards target
        # Compute the unit vector towards the target
        if distance > 0.005:
            desired_direction = position_error / distance
        else:
            desired_direction = np.zeros_like(position_error)

        # Compute the velocity towards the target
        velocity_towards_target = np.dot(linear_velocity, desired_direction)

        #only negative velocity is penalized
        #velocity_towards_target = np.clip(velocity_towards_target, -1000, 0)

        reward += rc["velocity_towards_target"] * velocity_towards_target
        reward_components["velocity_towards_target"] = (
            rc["velocity_towards_target"] * velocity_towards_target
        )

        # Linear velocity penalty
        # linear_vel_penalty = rc["linear_velocity"] * (
        #     np.linalg.norm(velocity_towards_target) - distance
        # )
        # reward -= linear_vel_penalty
        # reward_components["linear_velocity_penalty"] = -linear_vel_penalty

        # Goal bonus
        
        goal_bonus = (
            0.5 * rc["goal_bonus"] * np.exp(-(distance**2) / 0.08**2)
        )  
        # exact peek at position
        peak_bonus = 0.5 * rc["goal_bonus"] * np.exp(-(distance**2) / 0.005**2)
        # only give bonus if velocity is near zero
        peak_bonus *= np.exp(-np.linalg.norm(linear_velocity)**2 / 0.1**2)

        goal_bonus += peak_bonus
    
        # # # Move the target if good tracking
        # if distance < 0.01:
        #     if np.random.default_rng().uniform() < self.target_move_prob * np.exp(-(distance**2) / 0.01**2):
        #         # Move the target to a new random position
        #         self.target_position = self.np_random.uniform(
        #             low=self.workspace["low"] + 0.1,
        #             high=self.workspace["high"] - 0.1,
        #         )
        #         # Update the goal marker's position if applicable
        #         self.model.geom_pos[self.goal_geom_id] = self.target_position

        #         self.max_time += 10  # add more time to reach the target
        #         # goal_bonus += (
        #         #     100 * rc["goal_bonus"]
        #         # )  # add more bonus for reaching the target and to prevent policy avoiding it
        #         goal_bonus += 100 * rc["goal_bonus"]
        reward += goal_bonus
        reward_components["goal_bonus"] = goal_bonus
        

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

        # Payload penalties:
        if self.payload:
            #payload id
            payload_joint_id = self.model.body_jntadr[self.payload_body_id]
            payload_velocity = self.data.qvel[payload_joint_id : payload_joint_id + 3]

            # Compute payload velocity penalty
            payload_velocity_penalty = rc["payload_velocity"] * np.linalg.norm(payload_velocity)**2
            reward -= payload_velocity_penalty
            reward_components["payload_velocity"] = -payload_velocity_penalty

            # above payload reward see reward.ipynb
            quad_target_offset = position - self.target_position
            z_offset = quad_target_offset[2]
            xy_distance = np.linalg.norm(quad_target_offset[:2])
            above_payload_reward = rc["above_payload"]*  (.04-3*(z_offset - 0.34)**4 - 6*(xy_distance)**2)
            reward += above_payload_reward
            reward_components["above_payload_reward"] = above_payload_reward



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
        position_std_dev = 0.5 * self.randomness  # Standard deviation in meters
        random_position = self.np_random.normal(
            loc=self.target_position, scale=position_std_dev
        )
        # Clip the position to be within the workspace bounds
        random_position = np.clip(
            random_position, self.workspace["low"]+0.1, self.workspace["high"]-0.1
        )

        if self.target_mode == "payload" and self.payload:
            random_position[2] += 0.2 # Start 20cm above the target position

        self.data.qpos[:3] = random_position

        #randomize intertial properties around <inertial pos="0 0 0" mass="0.034" diaginertia="1.657171e-5 1.6655602e-5 2.9261652e-5"/>
        self.model.body_mass[self.drone_body_id] = np.clip(self.np_random.normal(loc=0.033, scale=0.03 * self.randomness), 0.025, 0.04)
        self.model.body_inertia[self.drone_body_id] = self.np_random.normal(loc=1, scale=0.1 * self.randomness) * np.array([1.657171e-5, 1.6655602e-5, 2.9261652e-5])

        # Randomize initial orientation around upright orientation
        orientation_std_dev = np.deg2rad(20) * self.randomness  # Standard deviation of 20 degrees
        roll = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        pitch = self.np_random.normal(loc=0.0, scale=orientation_std_dev)
        #clip roll and pitch to be within reasonable range
        max_deg = 90
        roll = np.clip(roll, -np.deg2rad(max_deg), np.deg2rad(max_deg))
        pitch = np.clip(pitch, -np.deg2rad(max_deg), np.deg2rad(max_deg))
      
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

        #randomize payload pos with negative z direction
        payload_direction = np.array([0, 0, -1])
        payload_direction[0:2] = self.np_random.normal(loc=0.0, scale=0.1 * self.randomness, size=2)
        payload_direction = payload_direction / np.linalg.norm(payload_direction)

        # scale to max cable length
        payload_offset = payload_direction * np.clip(np.random.normal(loc=0.18, scale=0.1 * self.randomness), 0.05, 0.18)
    
        print("payload offset", payload_offset)
        self.data.qpos[payload_qpos_index : payload_qpos_index + 3] = random_position + payload_offset

        #randomize payload mass from 1 to 11g
        self.model.body_mass[self.payload_body_id] = np.clip(self.np_random.normal(loc=0.005, scale=0.02 * self.randomness), 0.001, 0.011)

        # warmup sim to stabilize rope
        while self.data.time < self.warmup_time:
            self.do_simulation(np.zeros(4), 10)
            # reset qpos
            self.data.qpos[:3] = random_position
            self.data.qpos[3:7] = q
            # keep payload vel low
            # if self.data.qvel[payload_qpos_index : payload_qpos_index + 3].shape[0] == 3:
            #     self.data.qvel[payload_qpos_index : payload_qpos_index + 3] = np.zeros(3)

            # # also keep payload pos fixed for initial cable stabilization
            # if self.data.time < 0.5 * self.warmup_time:
            #     self.data.qpos[payload_qpos_index : payload_qpos_index + 3] = random_position + payload_offset
            

        self.warmup_time = self.data.time

        # Randomize velocity
        self.data.qvel[:3] = np.clip(self.np_random.normal(
            loc=0.0, scale=0.4 * self.randomness, size=3
        ), -1, 1) # Random linear velocity

        self.data.qvel[3:6] = np.clip(self.np_random.normal(
            loc=0.0, scale=0.1 * self.randomness, size=3
        ), -3, 3)

        #randomize max_thrust of motors
        self.max_thrust = self.np_random.uniform(low=0.08, high=0.14) * self.randomness + (1 - self.randomness)*0.11772
        self.motor_offset = self.np_random.normal(loc=1.0, scale=0.01 * self.randomness, size=4)
        

        # Randomize initial actions in the action space
        initial_action = self.action_space.sample()
        self.data.ctrl[:] = initial_action[:4]

        # Update the goal marker position
        self.model.geom_pos[self.goal_geom_id] = self.target_position

        mujoco.mj_forward(self.model, self.data)

        # Return initial observation
        obs = self._get_obs()
        return obs
