#!/usr/bin/env python
"""
multi_quad_mava_full.py

This file defines a MultiQuadEnv (a Brax/MJX version of a multi-rotor quadcopter team with payload)
and then wraps it for multi-agent decentralized MAPPO training using Mava. Each quad is treated as its
own agent. The privileged (centralized) critic network uses hidden layers [128, 128, 128, 128] and the
agent policies use hidden layers [128, 64, 64].

Before running, ensure that:
  - You have the required dependencies installed (jax, brax, mujoco, ml_collections, wandb, etc.)
  - The MuJoCo XML file "two_quad_payload.xml" is available at the specified path.
  - Mava (and its ecosystem libraries) is installed.

Usage:
    python multi_quad_mava_full.py
"""

import os
import functools
from datetime import datetime
import time
import imageio
import wandb

import jax
from jax import numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
from brax import envs, math
from brax.envs.base import PipelineEnv, State
import ml_collections  # for configuration management

# =============================================================================
# Original MultiQuadEnv definition (adapted from your training script)
# =============================================================================

# Helper functions used by MultiQuadEnv:
def jp_R_from_quat(q: jnp.ndarray) -> jnp.ndarray:
    """Compute rotation matrix from quaternion [w, x, y, z]."""
    q = q / jnp.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    r1 = jnp.array([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)])
    r2 = jnp.array([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)])
    r3 = jnp.array([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)])
    return jnp.stack([r1, r2, r3])

def jp_angle_between(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    norm1 = jnp.linalg.norm(v1)
    norm2 = jnp.linalg.norm(v2)
    dot = jnp.dot(v1, v2)
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    return jnp.arccos(cos_theta)

class MultiQuadEnv(PipelineEnv):
    """
    A Brax/MJX version of a multi-rotor quadcopter team with payload.
    This environment converts an original gym-like multi-quad setup into a format
    that trains policies with Brax (using MJX physics).
    """
    def __init__(self, policy_freq: float = 250, sim_steps_per_action: int = 1,
                 max_time: float = 10.0, reset_noise_scale: float = 0.2, **kwargs):
        # Load the MJX model from the XML file.
        mj_model = mujoco.MjModel.from_xml_path("two_quad_payload.xml")
        # Load the model using mjcf (assuming mjcf.load_model is available)
        from brax import mjcf  # make sure mjcf is in your PYTHONPATH
        sys = mjcf.load_model(mj_model)
        kwargs['n_frames'] = kwargs.get('n_frames', sim_steps_per_action)
        kwargs['backend'] = 'mjx'
        super().__init__(sys, **kwargs)
        
        self.policy_freq = policy_freq
        self.sim_steps_per_action = sim_steps_per_action
        self.time_per_action = 1.0 / self.policy_freq
        self.max_time = max_time
        self._reset_noise_scale = reset_noise_scale
        self.warmup_time = 1.0
        
        dt = self.time_per_action / self.sim_steps_per_action
        sys.mj_model.opt.timestep = dt
        
        self.max_thrust = 0.11772
        self.goal_center = jnp.array([0.0, 0.0, 1])
        self.goal_radius = 0.8
        self.target_position = self.goal_center
        
        self.payload_body_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
        self.q1_body_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_cf2")
        self.q2_body_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_cf2")
        self.goal_site_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "goal_marker")
    
    def reset(self, rng: jnp.ndarray) -> State:
        # Reset the environment state with added noise.
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        data = self.pipeline_init(qpos, qvel)
        last_action = jnp.zeros(self.sys.nu)
        metrics = {'time': data.time, 'reward': jnp.array(0.0)}
        obs = self._get_obs(data, last_action, self.target_position)
        reward = jnp.array(0.0)
        done = jnp.array(0.0)
        return State(data, obs, reward, done, metrics)
    
    def step(self, state: State, action: jnp.ndarray) -> State:
        prev_last_action = state.obs[-(self.sys.nu+1):-1]
        thrust_cmds = 0.5 * (action + 1.0)
        action_scaled = thrust_cmds * self.max_thrust
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action_scaled)
        target = self.target_position

        # Compute the tilt for each quad
        q1_orientation = data.xquat[self.q1_body_id]
        q2_orientation = data.xquat[self.q2_body_id]
        up = jnp.array([0.0, 0.0, 1.0])
        q1_local_up = jp_R_from_quat(q1_orientation)
        q2_local_up = jp_R_from_quat(q2_orientation)
        angle_q1 = jp_angle_between(q1_local_up[:,2], up)
        angle_q2 = jp_angle_between(q2_local_up[:,2], up)

        quad1_pos = data.xpos[self.q1_body_id]
        quad2_pos = data.xpos[self.q2_body_id]
        quad_distance = jnp.linalg.norm(quad1_pos - quad2_pos)
        collision = quad_distance < 0.11
        out_of_bounds = jnp.logical_or(jnp.abs(angle_q1) > jnp.radians(80),
                                       jnp.abs(angle_q2) > jnp.radians(80))
        out_of_bounds = jnp.logical_or(out_of_bounds, data.xpos[self.q1_body_id][2] < 0.05)
        out_of_bounds = jnp.logical_or(out_of_bounds, data.xpos[self.q2_body_id][2] < 0.05)
        out_of_bounds = jnp.logical_or(out_of_bounds, data.xpos[self.q1_body_id][2] < data.xpos[self.payload_body_id][2])
        out_of_bounds = jnp.logical_or(out_of_bounds, data.xpos[self.q2_body_id][2] < data.xpos[self.payload_body_id][2])
        out_of_bounds = jnp.logical_or(out_of_bounds, data.xpos[self.payload_body_id][2] < 0.05)

        obs = self._get_obs(data, prev_last_action, target)
        reward, _, _ = self.calc_reward(
            obs, data.time, collision, out_of_bounds, action_scaled,
            angle_q1, angle_q2, prev_last_action, target
        )
        done = jnp.logical_or(out_of_bounds, collision)
        done = jnp.logical_or(done, data.time > self.max_time) * 1.0
        new_metrics = {'time': data.time, 'reward': reward}
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=new_metrics)
    
    def _get_obs(self, data, last_action: jnp.ndarray, target_position) -> jnp.ndarray:
        # Construct the observation vector.
        payload_pos = data.xpos[self.payload_body_id]
        payload_linvel = data.cvel[self.payload_body_id][3:6]
        payload_error = target_position - payload_pos

        quad1_pos = data.xpos[self.q1_body_id]
        quad1_quat = data.xquat[self.q1_body_id]
        quad1_linvel = data.cvel[self.q1_body_id][3:6]
        quad1_angvel = data.cvel[self.q1_body_id][:3]
        quad1_rel = quad1_pos - payload_pos
        quad1_rot = jp_R_from_quat(quad1_quat).ravel()
        quad1_linear_acc = data.cacc[self.q1_body_id][3:6]
        quad1_angular_acc = data.cacc[self.q1_body_id][:3]

        quad2_pos = data.xpos[self.q2_body_id]
        quad2_quat = data.xquat[self.q2_body_id]
        quad2_linvel = data.cvel[self.q2_body_id][3:6]
        quad2_angvel = data.cvel[self.q2_body_id][:3]
        quad2_rel = quad2_pos - payload_pos
        quad2_rot = jp_R_from_quat(quad2_quat).ravel()
        quad2_linvel = data.cvel[self.q2_body_id][3:6]
        quad2_angvel = data.cvel[self.q2_body_id][:3]
        quad2_linear_acc = data.cacc[self.q2_body_id][3:6]
        quad2_angular_acc = data.cacc[self.q2_body_id][:3]

        time_progress = jnp.array([(data.time - self.warmup_time) / self.max_time])
        obs = jnp.concatenate([
            payload_error,        # 3 elements
            payload_linvel,       # 3 elements
            quad1_rel,            # 3 elements
            quad1_rot,            # 9 elements
            quad1_linvel,         # 3 elements
            quad1_angvel,         # 3 elements
            quad1_linear_acc,     # 3 elements
            quad1_angular_acc,    # 3 elements
            quad2_rel,            # 3 elements
            quad2_rot,            # 9 elements
            quad2_linvel,         # 3 elements
            quad2_angvel,         # 3 elements
            quad2_linear_acc,     # 3 elements
            quad2_angular_acc,    # 3 elements
            last_action,          # (nu elements)
        ])
        return obs

    def calc_reward(self, obs, sim_time, collision, out_of_bounds, action,
                    angle_q1, angle_q2, last_action, target_position):
        # Split observation into team and quad components.
        team_obs = obs[:6]
        quad1_obs = obs[6:30]
        quad2_obs = obs[30:54]
        quad_distance = jnp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])
        
        payload_error = team_obs[:3]
        payload_linvel = team_obs[3:6]
        linvel_penalty = jnp.linalg.norm(payload_linvel)
        dis = jnp.linalg.norm(payload_error)
        z_error = jnp.abs(payload_error[2])
        distance_reward = (1 - dis + jnp.exp(-10 * dis)) + jnp.exp(-10 * z_error)
        
        norm_error = jnp.maximum(jnp.linalg.norm(payload_error), 1e-6)
        norm_linvel = jnp.maximum(jnp.linalg.norm(payload_linvel), 1e-6)
        velocity_towards_target = jnp.dot(payload_error, payload_linvel) / (norm_error * norm_linvel)
      
        safe_distance_reward = jnp.clip((quad_distance - 0.11) / (0.15 - 0.11), 0, 1)
        collision_penalty = 5.0 * collision
        out_of_bounds_penalty = 50.0 * out_of_bounds
        smooth_action_penalty = jnp.mean(jnp.abs(action - last_action) / self.max_thrust)
        action_energy_penalty = jnp.mean(jnp.abs(action)) / self.max_thrust
        
        quad1_rel = quad1_obs[:3]
        quad2_rel = quad2_obs[:3]
        z_reward_q1 = quad1_rel[2] - target_position[2]
        z_reward_q2 = quad2_rel[2] - target_position[2]
        quad_above_reward = z_reward_q1 + z_reward_q2
        
        up_reward = jnp.exp(-jnp.abs(angle_q1)) + jnp.exp(-jnp.abs(angle_q2))
        
        ang_vel_q1 = quad1_obs[15:18]
        ang_vel_q2 = quad2_obs[15:18]
        ang_vel_penalty = 0.1 * (jnp.linalg.norm(ang_vel_q1)**2 + jnp.linalg.norm(ang_vel_q2)**2)
        linvel_q1 = quad1_obs[9:12]
        linvel_q2 = quad2_obs[9:12]
        linvel_quad_penalty = 0.1 * (jnp.linalg.norm(linvel_q1)**2 + jnp.linalg.norm(linvel_q2)**2)
        
        reward = 0
        reward += 10 * distance_reward 
        reward += safe_distance_reward
        reward += velocity_towards_target
        reward += up_reward
        reward -= 10 * linvel_penalty
        reward -= collision_penalty
        reward -= out_of_bounds_penalty
        reward -= 2 * smooth_action_penalty
        reward -= action_energy_penalty
        reward -= ang_vel_penalty
        reward -= 5 * linvel_quad_penalty
        
        reward /= 25.0
        return reward, None, {}

# Register the environment so that brax.envs.get_environment('multiquad') returns an instance.
envs.register_environment('multiquad', MultiQuadEnv)

# =============================================================================
# Multi-Agent Wrapper for MAPPO (each quad is its own agent)
# =============================================================================

class MultiAgentQuadEnvWrapper:
    """
    Wraps the MultiQuadEnv so that its reset() and step() functions return a dictionary mapping:
      - 'quad1': observation for quad1
      - 'quad2': observation for quad2
    For this example, we assume that _get_obs() returns a vector organized as:
      [team_info (6), quad1_info (24), quad2_info (24), last_action (nu)]
    """
    def __init__(self, env):
        self.env = env
        self.agent_ids = ['quad1', 'quad2']
        # Set dimensions (adjust these if your env observation layout differs)
        self.team_dim = 6
        self.quad_info_dim = 24
        self.last_action_dim = env.sys.nu  # last nu entries

    def reset(self, rng):
        state = self.env.reset(rng)
        obs = state.obs
        team_info = obs[:self.team_dim]
        quad1_info = obs[self.team_dim:self.team_dim + self.quad_info_dim]
        quad2_info = obs[self.team_dim + self.quad_info_dim:self.team_dim + 2 * self.quad_info_dim]
        last_action = obs[self.team_dim + 2 * self.quad_info_dim:]
        obs_dict = {
            'quad1': jnp.concatenate([team_info, quad1_info, last_action]),
            'quad2': jnp.concatenate([team_info, quad2_info, last_action])
        }
        return obs_dict, state

    def step(self, state, actions):
        # actions is a dict: {'quad1': action1, 'quad2': action2}
        # For simplicity, we combine the two agents' actions by averaging.
        action_quad1 = jnp.array(actions['quad1'])
        action_quad2 = jnp.array(actions['quad2'])
        combined_action = 0.5 * (action_quad1 + action_quad2)
        new_state = self.env.step(state, combined_action)
        obs = new_state.obs
        team_info = obs[:self.team_dim]
        quad1_info = obs[self.team_dim:self.team_dim + self.quad_info_dim]
        quad2_info = obs[self.team_dim + self.quad_info_dim:self.team_dim + 2 * self.quad_info_dim]
        last_action = obs[self.team_dim + 2 * self.quad_info_dim:]
        obs_dict = {
            'quad1': jnp.concatenate([team_info, quad1_info, last_action]),
            'quad2': jnp.concatenate([team_info, quad2_info, last_action])
        }
        rewards = {
            'quad1': new_state.reward,
            'quad2': new_state.reward
        }
        return obs_dict, new_state, rewards

# =============================================================================
# Mava Experiment Setup for Decentralized MAPPO
# =============================================================================

def make_multiagent_env(config):
    """
    Creates and returns a multi-agent environment using the registered 'multiquad' environment,
    wrapped with the MultiAgentQuadEnvWrapper.
    """
    env = envs.get_environment('multiquad')
    return MultiAgentQuadEnvWrapper(env)

def main():
    # Import Mava experiment runner and configuration helper.
    from mava.systems.ppo.anakin.ff_mappo import run_experiment
    from mava.utils.configurations import get_default_config

    # Get default configuration for decentralized MAPPO.
    config = get_default_config("mappo_decentralized")
    
    # Update network architectures:
    #   - Agent policies: [128, 64, 64]
    #   - Privileged critic: [128, 128, 128, 128]
    config["network"] = {
        "policy_hidden_sizes": [128, 64, 64],
        "critic_hidden_sizes": [128, 128, 128, 128]
    }
    
    # Define agent IDs for the two quads.
    config["agents"] = ["quad1", "quad2"]
    
    # Set environment configuration (optional).
    config["environment"] = {"env_name": "multiquad"}
    
    # (Optional) Adjust other hyperparameters as needed.
    # e.g., config["num_timesteps"] = 250_000_000

    # Run the experiment.
    run_experiment(config=config, env_creator=make_multiagent_env)

if __name__ == "__main__":
    main()