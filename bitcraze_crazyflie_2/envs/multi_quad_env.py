import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import time  # added for debugging
import numba
from numba import njit

from typing import Dict, Union
from scipy.spatial.transform import Rotation as R
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.0,
}

MAX_THRUST = 0.11772



@njit
def np_R_from_quat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    r = np.empty((3,3))
    r[0, 0] = 1 - 2*(y*y + z*z)
    r[0, 1] = 2*(x*y - z*w)
    r[0, 2] = 2*(x*z + y*w)
    r[1, 0] = 2*(x*y + z*w)
    r[1, 1] = 1 - 2*(x*x + z*z)
    r[1, 2] = 2*(y*z - x*w)
    r[2, 0] = 2*(x*z - y*w)
    r[2, 1] = 2*(y*z + x*w)
    r[2, 2] = 1 - 2*(x*x + y*y)
    return r

@njit
def np_to_frame(q, vec):
    r = np_R_from_quat(q)
    # For an orthonormal rotation matrix, the inverse is the transpose
    return r.T @ vec

@njit
def np_angle_between(v1, v2):
    norm1 = 0.0
    norm2 = 0.0
    for i in range(v1.shape[0]):
        norm1 += v1[i] * v1[i]
        norm2 += v2[i] * v2[i]
    norm1 = norm1 ** 0.5
    norm2 = norm2 ** 0.5
    dot = 0.0
    for i in range(v1.shape[0]):
        dot += v1[i] * v2[i]
    # Avoid division by zero.
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_theta = dot / (norm1 * norm2)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    return np.arccos(cos_theta)

@njit
def np_to_spherical(vec):
    r = np.linalg.norm(vec)
    theta = np.arctan2(vec[1], vec[0])
    phi = np.arccos(vec[2] / r)
    return np.array([r, theta, phi])


class MultiQuadEnv(MujocoEnv):
    def __init__(
        self,
        reward_coefficients=None,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        policy_freq=250,  # Policy frequency in Hz
        sim_steps_per_action=3,  # Simulation steps between policy executions
        render_mode=None,
        visual_options=None,
        env_config={},
        target_move_prob=0.01,  # Probability of target moving when drone reaches it
        **kwargs,
    ):
        # Path to the MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), "mujoco", "two_quad_payload.xml")


        self.randomness = 0#env_config.get("randomness", 0.1)
        self.obs_buffer_size = 1
 


        self.debug_rates_enabled = env_config.get("debug_rates_enabled", False)

        self.debug_rates = {'sim': [], 'obs': [], 'reward': [], 'total': []}

       

        self.DEFAULT_CAMERA_CONFIG = default_camera_config

        # Set parameters
        self.policy_freq = policy_freq
        self.sim_steps_per_action = sim_steps_per_action

        # Compute time per action
        self.time_per_action = 1.0 / self.policy_freq  # Time between policy executions

        # Set frame_skip to sim_steps_per_action
        frame_skip = self.sim_steps_per_action

        self.max_time = env_config.get("max_time", 10.0)


        self.warmup_time = 1.0  # 1s warmup time


    
   

        base_obs_dim = 63


        

        

        
        obs_low = np.full(base_obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(base_obs_dim, np.inf, dtype=np.float32)
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

        
        self.sum_coefficients = sum(self.reward_coefficients.values())

    

        # Set the target move probability
        self.target_move_prob = target_move_prob


    
        self.current_thrust = np.zeros(8)


    

    def R_from_quat(self, q):
        # Use Numba-optimized quaternion conversion
        return np_R_from_quat(q)
    
    def to_frame(self, orientation_q, vec):
        # Use Numba-optimized vector rotation
        return np_to_frame(orientation_q, vec)
    
    def to_spherical(self, vec):
        # Use Numba-optimized vector rotation
        return np_to_spherical(vec)



    def _get_obs(self):
        # Get payload state via named lookup 
        payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        payload_pos = self.data.xpos[payload_body_id].copy()
        payload_linvel = self.data.cvel[payload_body_id].copy()[3:6]  # Extract linear part from cvel (indices 3:6)
        payload_error = self.target_position - payload_pos

        # Get quad 1 state via named lookup
        quad1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q0_cf2")
        quad1_pos = self.data.xpos[quad1_body_id].copy()
        quad1_quat = self.data.xquat[quad1_body_id].copy()  # Quaternion [w, x, y, z]
        quad1_linvel = self.data.cvel[quad1_body_id].copy()[3:6] 
        quad1_angvel = self.data.cvel[quad1_body_id].copy()[:3]         
        quad1_rel = quad1_pos - payload_pos
        quad1_rot = self.R_from_quat(quad1_quat).flatten()  # 9 elements
        quad1_linear_acc = self.data.cacc[quad1_body_id][3:6]
        quad1_angular_acc = self.data.cacc[quad1_body_id][:3]

        # Get quad 2 state via named lookup
        quad2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q1_cf2")
        quad2_pos = self.data.xpos[quad2_body_id].copy()
        quad2_quat = self.data.xquat[quad2_body_id].copy()  # Quaternion [w, x, y, z]
        quad2_linvel = self.data.cvel[quad2_body_id].copy()[3:6] 
        quad2_angvel = self.data.cvel[quad2_body_id].copy()[:3]     
        quad2_rel = quad2_pos - payload_pos
        quad2_rot = self.R_from_quat(quad2_quat).flatten()  # 9 elements
        quad2_linear_acc = self.data.cacc[quad2_body_id][3:6]
        quad2_angular_acc = self.data.cacc[quad2_body_id][:3]

        # # Get sensor readings (accelerometer and gyro for both quads)
        # def get_sensor(sensor_name):
        #     sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        #     adr = self.model.sensor_adr[sensor_id]
        #     dim = self.model.sensor_dim[sensor_id]
        #     return self.data.sensordata[adr:adr + dim].copy()

        # quad1_gyro = get_sensor("q0_gyro")
        # quad1_acc = get_sensor("q0_linacc")
        # quad2_gyro = get_sensor("q1_gyro")
        # quad2_acc = get_sensor("q1_linacc")

        # Retrieve last_action (default to zeros if not set)
        last_action = self.last_action if hasattr(self, "last_action") else np.zeros(8, dtype=np.float32)

        # Time progress for value function
        time_progress = np.array([(self.data.time - self.warmup_time) / self.max_time])

        # Build observation vector:
        #   payload_error (3), payload_linvel (3),
        #   quad1: relative position (3), rotation matrix flatten (9), linear velocity (3), angular velocity (3), linear acceleration (3), angular acceleration (3),
        #   quad2: relative position (3), rotation matrix flatten (9), linear velocity (3), angular velocity (3), linear acceleration (3), angular acceleration (3),
        #   last_action (8) and time_progress (1)
        obs = np.concatenate([
            payload_error,        # (3,)
            payload_linvel,       # (3,)
            quad1_rel,            # (3,)
            quad1_rot,            # (9,)
            quad1_linvel,         # (3,)
            quad1_angvel,         # (3,)
            quad1_linear_acc,     # (3,)
            quad1_angular_acc,    # (3,)
            quad2_rel,            # (3,)
            quad2_rot,            # (9,)
            quad2_linvel,         # (3,)
            quad2_angvel,         # (3,)
            quad2_linear_acc,     # (3,)
            quad2_angular_acc,    # (3,)
            last_action,          # (8,)
            time_progress         # (1,)
        ])
        
        # Total dims = 3+3 + (3+9+3+3+3+3)*2 + 8 + 1 = 63
        obs = self.noise_observation(obs, noise_level=0.05)
        return obs.astype(np.float32)

    def noise_observation(self, obs, noise_level=0.02):

        obs *= np.random.normal(loc=0, scale=noise_level, size=obs.shape)
        return obs
    
    def _stack_obs(self):
        # Concatenate observations from the buffer by sampling every 'stack_stride' element in reverse order (most recent first)
        indices = range(len(self.obs_buffer) - 1, -1, -self.stack_stride)
        stacked = np.concatenate([self.obs_buffer[i] for i in indices])
        return stacked

    def step(self, action):
        sim_time = self.data.time
        

        # Convert action from [-1,1] to [0,1]
        thrust_cmds = 0.5 * (action + 1.0)
       
        #throw error if action is out of bounds
        if np.any(thrust_cmds < 0) or np.any(thrust_cmds > 1):
            raise ValueError("Action out of bounds")

        action_scaled = thrust_cmds * self.max_thrust


        self.current_thrust = action_scaled

     

        # Store last action
        self.last_action = (self.data.ctrl[:8].copy() / self.max_thrust) * 2.0 - 1.0
        
    
        
        # Run simulation
        
        self.do_simulation(self.current_thrust, self.frame_skip)
       
        
        # Get observation

        obs = self._get_obs()

        if (self.obs_buffer_size > 1):
            self.obs_buffer.append(obs)
        
            if len(self.obs_buffer) > self.obs_buffer_size:
                self.obs_buffer.pop(0)

            stacked_obs = self._stack_obs()
        else:
            stacked_obs = obs


        # Check for collisions
        collision = False

        quad_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q0_cf2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q1_cf2"),
        ]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
 
            if body1 in quad_ids and body2 in quad_ids:
                collision = True
                break

        # Check if out of bounds
        
        out_of_bounds = False
        q1_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q0_cf2")]
        q2_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q1_cf2")]
        positions = [q1_pos, q2_pos]
        for position in positions:
            out_of_bounds = not np.all(self.workspace["low"] <= position) or not np.all(
                position <= self.workspace["high"]
            )
            if out_of_bounds:
                break

        # Constraint orientation to 90 degrees
        # Global up vector in world frame
        up = np.array([0, 0, 1])

        # Get the orientations (quaternions) for each quad
        q1_orientation = self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q0_cf2")]
        q2_orientation = self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q1_cf2")]

        # Convert quaternions to rotation matrices and extract local up vectors (third column)
        q1_local_up = self.R_from_quat(q1_orientation)[:, 2]
        q2_local_up = self.R_from_quat(q2_orientation)[:, 2]

        # Compute the tilt angle for each quad (0 means perfectly horizontal)
        angle_q1 = self.angle_between(q1_local_up, up)
        angle_q2 = self.angle_between(q2_local_up, up)

       

        # Check if either quad's orientation deviates more than 90 degrees:
        if angle_q1 > np.pi / 2 or angle_q2 > np.pi / 2:
            out_of_bounds = True
            #print("Quad orientation exceeded 90 degrees from horizontal")
        


        # limit velocity to 2m/s
        q1_vel = self.data.cvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q0_cf2")]
        q2_vel = self.data.cvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "q1_cf2")]
        
       
        if np.linalg.norm(q1_vel) > 15 or np.linalg.norm(q2_vel) > 15:
            out_of_bounds = True
            #print("Velocity out of bounds", np.linalg.norm(q1_vel), np.linalg.norm(q2_vel))
    

        

        if not np.all(np.isfinite(self.data.qacc)):
            # Simulation gets unstable because of fast movements
            out_of_bounds = True
            #print("Simulation unstable")

        terminated = False
        
         # Terminate if going away from target again
        payload_joint_id = self.model.body_jntadr[self.payload_body_id]
        payload_pos = self.data.qpos[payload_joint_id : payload_joint_id + 3]
        position_error = self.target_position - payload_pos
        
        distance = np.linalg.norm(position_error)

        max_delta_distance = 1.5
        #lower max delta wiht time
        current_time_progress = (sim_time - self.warmup_time) / self.max_time
        min_delta_distance = (0.3+ 0.2*self.randomness) * (1.1-current_time_progress)

        if distance > max(self.max_distance * max_delta_distance, min_delta_distance):
            terminated = True
            out_of_bounds = True

        # Compute reward
        reward, reward_components, additional_info = self.calc_reward(
            obs,
            sim_time,
            collision,
            out_of_bounds,
            action_scaled,
            angle_q1, 
            angle_q2,
            last_action= self.last_action_scaled if hasattr(self, "last_action_scaled") else action_scaled,
        )
      

        self.last_action_scaled = action_scaled
        # Determine termination conditions
        terminated = terminated or collision or out_of_bounds
        truncated = False

        if distance < self.max_distance:
            self.max_distance = distance


        # Truncate episode if too long
        if (
            sim_time - self.warmup_time > self.max_time
        ):
            terminated = True
            truncated = True

        # Additional info
        info = {
            "position": position.copy(),
         
            "reward_components": reward_components,
            "action": action_scaled,
        }
        info.update(additional_info)

        
        # if self.render_mode == "human":
        #     self.render()


        return stacked_obs, reward, terminated, truncated, info
    
    def angle_between(self, v1, v2):
        # Call the njit-optimized helper.
        return np_angle_between(np.array(v1), np.array(v2))
    

    def calc_team_reward(self, team_obs, quad_distance, sim_time, collision, out_of_bounds, action, last_action):
        payload_error = team_obs[:3]
        payload_vel = team_obs[3:6]


        # Compute distance to target position
        distance_penalty = np.linalg.norm(10*payload_error)*sim_time

        # Velocity towards target
        velocity_towards_target = np.dot(payload_error, payload_vel) / (np.linalg.norm(payload_error) * np.linalg.norm(payload_vel) + 1e-6)
        
        # Safe distance of quads reward as gaussian
        safe_distance_penalty = np.exp(-0.5 * (quad_distance - 0.5)**2 / 0.1**2)

        # Collision and out of bounds penalties
        collision_penalty = 1 if collision else 0
        out_of_bounds_penalty = 1 if out_of_bounds else 0

        # Smooth action penalty
        if hasattr(self, "last_action"):
            smooth_action_penalty = np.mean(
                np.abs(action - last_action)/self.max_thrust
            )/10
        else:
            smooth_action_penalty = 0

        return distance_penalty, velocity_towards_target, safe_distance_penalty, collision_penalty, out_of_bounds_penalty, smooth_action_penalty
        
        
    

    
    def calc_quad_reward(self, quad_obs, angle):
        quad_rel = quad_obs[:3]
        # quad_rot = quad_obs[3:12]  # unused for penalty computation
        linvel = quad_obs[12:15]
        angvel = quad_obs[15:18]            
        linacc = quad_obs[18:21]
        angacc = quad_obs[21:24]

        rotation_penalty = np.abs(angle)
        linear_velocity_penalty = np.linalg.norm(linvel)
        angular_velocity_penalty = np.linalg.norm(angvel) 
        linear_acc_penalty = np.linalg.norm(linacc)
        angular_acc_penalty = np.linalg.norm(angacc)
        above_payload_penalty = -quad_rel[2]
        return rotation_penalty, angular_velocity_penalty, linear_velocity_penalty, linear_acc_penalty, angular_acc_penalty, above_payload_penalty

    def calc_reward(
        self, obs, sim_time, collision, out_of_bounds, action, angle_q1, angle_q2, last_action
    ):
        
        team_obs = obs[:6]
        quad1_obs = obs[6:30]
        quad2_obs = obs[30:54]

        quad_distance = np.linalg.norm(quad1_obs[:3] - quad2_obs[:3])
        

        distance_penalty, velocity_towards_target, safe_distance_penalty, collision_penalty, out_of_bounds_penalty, smooth_action_penalty = self.calc_team_reward(team_obs, quad_distance, sim_time, collision, out_of_bounds, action, last_action)
        quad1_reward = self.calc_quad_reward(quad1_obs, angle_q1)
        quad2_reward = self.calc_quad_reward(quad2_obs, angle_q2)

        # Unpack quad rewards: each returns (rotation, angular_velocity, linacc, angacc, above_payload)
        rotation_penalty = quad1_reward[0] + quad2_reward[0]
        angular_velocity_penalty = quad1_reward[1] + quad2_reward[1]
        linear_velocity_penalty = quad1_reward[2] + quad2_reward[2]
        linear_acc_penalty = quad1_reward[3] + quad2_reward[3]
        angular_acc_penalty = quad1_reward[4] + quad2_reward[4]
        above_payload_penalty = quad1_reward[5] + quad2_reward[5]

        reward_components = {}
        reward_components["alive_reward"] = 3
        reward = 3
        
        reward_components["distance_penalty"] = -distance_penalty
        reward += -distance_penalty
        reward_components["velocity_towards_target"] = velocity_towards_target
        reward += velocity_towards_target
        reward_components["safe_distance_penalty"] = -safe_distance_penalty
        reward += -safe_distance_penalty
        reward_components["collision_penalty"] = -collision_penalty
        reward += -collision_penalty
        reward_components["out_of_bounds_penalty"] = -out_of_bounds_penalty
        reward += -out_of_bounds_penalty
        reward_components["rotation_penalty"] = -rotation_penalty
        reward += -rotation_penalty
        reward_components["angular_velocity_penalty"] = -angular_velocity_penalty
        reward += -angular_velocity_penalty
        reward_components["linear_velocity_penalty"] = -linear_velocity_penalty
        reward += -linear_velocity_penalty
        reward_components["linear_acc_penalty"] = -linear_acc_penalty
        reward += -linear_acc_penalty
        reward_components["angular_acc_penalty"] = -angular_acc_penalty
        reward += -angular_acc_penalty
        reward_components["above_payload_penalty"] = -above_payload_penalty
        reward += -above_payload_penalty
        reward_components["smooth_action_penalty"] = -smooth_action_penalty
        reward += -smooth_action_penalty



        #frequency normalize
        reward /= self.time_per_action*1000

        # Additional info
        additional_info = {
            "rotation_penalty": rotation_penalty,
            "distance_to_target": distance_penalty,
        }

        return reward, reward_components, additional_info

    def reset_model(self):

        #reset mujoco model
        mujoco.mj_resetData(self.model, self.data)

        # Reset the simulation time
        self.data.time = 0.0

        # warmup sim to stabilize rope
        while self.data.time < self.warmup_time:
            self.do_simulation(np.zeros(8), 10)
        
        # disable equality constraints
        self.data.eq_active[:] = 0
            

        self.warmup_time = self.data.time

        
        #randomize max_thrust of motors
        self.max_thrust = np.clip(self.np_random.normal(loc=MAX_THRUST, scale=0.01 * self.randomness), 0.095, 0.13)
        self.motor_offset = self.np_random.normal(loc=1.0, scale=0.05 * self.randomness, size=4)

        

        # Randomize initial actions in the action space
        initial_action = self.action_space.sample()

        # scale actions
        thrust_cmds = 0.5 * (initial_action + 1.0)
        initial_action_scaled = thrust_cmds * self.max_thrust

        # Initialize last_action for future steps.
        self.last_action = initial_action_scaled.copy()


        # Update the goal marker position
        self.model.geom_pos[self.goal_geom_id] = self.target_position


        # step simulation to get initial observation
        self.do_simulation(initial_action_scaled, self.frame_skip)

        # Return initial observation
        obs = self._get_obs()


        self.max_distance = 5

        return obs