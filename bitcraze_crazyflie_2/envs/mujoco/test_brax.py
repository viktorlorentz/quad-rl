from datetime import datetime
import functools

from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

jax.config.update('jax_platform_name', 'gpu')

# ----------------------------------------
# Helper functions in JAX (converted from numpy/numba)
def jp_R_from_quat(q: jp.ndarray) -> jp.ndarray:
  """Compute rotation matrix from quaternion [w, x, y, z]."""
  q = q / jp.linalg.norm(q)
  w, x, y, z = q[0], q[1], q[2], q[3]
  r1 = jp.array([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)])
  r2 = jp.array([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)])
  r3 = jp.array([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)])
  return jp.stack([r1, r2, r3])

def jp_angle_between(v1: jp.ndarray, v2: jp.ndarray) -> jp.ndarray:
  norm1 = jp.linalg.norm(v1)
  norm2 = jp.linalg.norm(v2)
  dot = jp.dot(v1, v2)
  cos_theta = dot / (norm1 * norm2 + 1e-6)
  cos_theta = jp.clip(cos_theta, -1.0, 1.0)
  return jp.arccos(cos_theta)

# ----------------------------------------

class MultiQuadEnv(PipelineEnv):
  """
  A Brax/MJX version of a multi-rotor quadcopter team with payload.
  This environment converts the original gym version (using MuJoCo python bindings)
  into a format that trains policies with Brax (using MJX physics).
  """
  def __init__(
      self,
      policy_freq: float = 250,              # Policy frequency in Hz.
      sim_steps_per_action: int = 3,           # Physics steps between control actions.
      max_time: float = 10.0,                  # Maximum simulation time per episode.
      reset_noise_scale: float = 1e-2,
      **kwargs,
  ):
    # Load the MJX model from the XML file.
    mj_model = mujoco.MjModel.from_xml_path("two_quad_payload.xml")
    # Adjust solver options similar to the Humanoid example.
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)
    kwargs['n_frames'] = kwargs.get('n_frames', sim_steps_per_action)
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)

    # Save parameters.
    self.policy_freq = policy_freq
    self.sim_steps_per_action = sim_steps_per_action
    self.time_per_action = 1.0 / self.policy_freq
    self.max_time = max_time
    self._reset_noise_scale = reset_noise_scale
    self.warmup_time = 1.0

    # Maximum thrust from original env.
    self.max_thrust = 0.11772
    # Set the target payload position.
    self.target_position = jp.array([0.0, 0.0, 1.5])
    
    # Cache body/geom IDs for faster lookup.
    self.payload_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.q1_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_cf2")
    self.q2_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_cf2")
    self.goal_geom_id = mujoco.mj_name2id(
        sys.mj_model,  mujoco.mjtObj.mjOBJ_GEOM.value, "goal_marker")
  
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
    data = self.pipeline_init(qpos, qvel)
    # Initialize last_action as zeros.
    last_action = jp.zeros(self.sys.nu)
    # Compute observation including last_action.
    obs = self._get_obs(data, last_action)
    reward = jp.array(0.0)
    done = jp.array(0.0)
    # Metrics should only include scalar values.
    metrics = {'time': data.time, 'reward': jp.array(0.0)}
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Advances the environment by one control step."""
    # Extract the previous last_action from the observation.
    # Recall that _get_obs appends last_action as the penultimate segment.
    prev_last_action = state.obs[-(self.sys.nu+1):-1]
    # Convert actions from [-1, 1] to thrust commands in [0, max_thrust]
    thrust_cmds = 0.5 * (action + 1.0)
    action_scaled = thrust_cmds * self.max_thrust

    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action_scaled)

    # Compute the tilt (angle from vertical) for each quad.
    q1_orientation = data.xquat[self.q1_body_id]
    q2_orientation = data.xquat[self.q2_body_id]
    up = jp.array([0.0, 0.0, 1.0])
    q1_local_up = jp_R_from_quat(q1_orientation)[:, 2]
    q2_local_up = jp_R_from_quat(q2_orientation)[:, 2]
    angle_q1 = jp_angle_between(q1_local_up, up)
    angle_q2 = jp_angle_between(q2_local_up, up)

    # For this conversion we assume collision and out-of-bounds flags are False.
    collision = False
    out_of_bounds = False

    # Compute new observation using the previous last_action.
    obs = self._get_obs(data, prev_last_action)
    reward, _, _ = self.calc_reward(
        obs, data.time, collision, out_of_bounds, action_scaled, angle_q1, angle_q2, prev_last_action)

    # Terminate if simulation time exceeds max_time.
    done = jp.where(data.time > self.max_time, 1.0, 0.0)
    new_metrics = {'time': data.time, 'reward': reward}
    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=new_metrics)

  def _get_obs(self, data, last_action: jp.ndarray) -> jp.ndarray:
    """Constructs the observation vector from simulation data."""
    # Payload state.
    payload_pos = data.xpos[self.payload_body_id]
    payload_linvel = data.cvel[self.payload_body_id][3:6]
    payload_error = self.target_position - payload_pos

    # Quad 1 state.
    quad1_pos = data.xpos[self.q1_body_id]
    quad1_quat = data.xquat[self.q1_body_id]
    quad1_linvel = data.cvel[self.q1_body_id][3:6]
    quad1_angvel = data.cvel[self.q1_body_id][:3]
    quad1_rel = quad1_pos - payload_pos
    quad1_rot = jp_R_from_quat(quad1_quat).ravel()
    quad1_linear_acc = data.cacc[self.q1_body_id][3:6]
    quad1_angular_acc = data.cacc[self.q1_body_id][:3]

    # Quad 2 state.
    quad2_pos = data.xpos[self.q2_body_id]
    quad2_quat = data.xquat[self.q2_body_id]
    quad2_linvel = data.cvel[self.q2_body_id][3:6]
    quad2_angvel = data.cvel[self.q2_body_id][:3]
    quad2_rel = quad2_pos - payload_pos
    quad2_rot = jp_R_from_quat(quad2_quat).ravel()
    quad2_linear_acc = data.cacc[self.q2_body_id][3:6]
    quad2_angular_acc = data.cacc[self.q2_body_id][:3]

    # Include last action and time progress.
    time_progress = jp.array([(data.time - self.warmup_time) / self.max_time])
    obs = jp.concatenate([
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
        last_action,          # (nu,) — appended as the penultimate segment
        time_progress         # (1,)
    ])
    return obs

  def calc_reward(self, obs, sim_time, collision, out_of_bounds, action, angle_q1, angle_q2, last_action):
    """
    Computes a reward similar to the original gym env by splitting the observation into team and quad parts.
    """
    # Team observation: payload error (3) and payload linear velocity (3).
    team_obs = obs[:6]
    # Quad observations: next 24 elements for quad1 and following 24 for quad2.
    quad1_obs = obs[6:30]
    quad2_obs = obs[30:54]
    quad_distance = jp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])

    payload_error = team_obs[:3]
    payload_linvel = team_obs[3:6]
    distance_penalty = jp.linalg.norm(10 * payload_error) * (1 + sim_time)
    velocity_towards_target = 10 * (jp.dot(payload_error, payload_linvel) /
                                    (jp.linalg.norm(payload_error) * jp.linalg.norm(payload_linvel) + 1e-6))
    safe_distance_penalty = jp.exp(-0.5 * ((quad_distance - 0.5) ** 2) / (0.1 ** 2))
    collision_penalty = 10.0 if collision else 0.0
    out_of_bounds_penalty = 10.0 if out_of_bounds else 0.0
    smooth_action_penalty = jp.mean(jp.abs(action - last_action) / self.max_thrust) / 10.0
    thrust_reward = jp.sum(action) * 10.0

    # Combine components to form the final reward.
    reward = 10.0
    reward -= distance_penalty
    reward += velocity_towards_target
    reward -= safe_distance_penalty
    reward -= collision_penalty
    reward -= out_of_bounds_penalty
    reward -= jp.abs(angle_q1) + jp.abs(angle_q2)
    reward -= smooth_action_penalty
    reward += thrust_reward

    reward = reward / (self.time_per_action * 1000)
    return reward, None, {"rotation_penalty": jp.abs(angle_q1) + jp.abs(angle_q2),
                           "distance_to_target": distance_penalty}

# Register the environment under the name 'multiquad'
envs.register_environment('multiquad', MultiQuadEnv)

# ----------------------------------------
# Visualize a Rollout and Train the Policy
# ----------------------------------------

env_name = 'multiquad'
env = envs.get_environment(env_name)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]
for i in range(10):
  ctrl = -0.1 * jp.ones(env.sys.nu)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

# media.show_video(env.render(rollout), fps=1.0 / env.dt)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=30_000_000,      # Give the agent enough interactions to learn complex dynamics.
    num_evals=10,                  # Evaluate frequently to monitor performance.
    reward_scaling=10,             # Scale rewards so that the gradients are well behaved; adjust if your rewards are very small or large.
    episode_length=1000,           # Allow each episode a fixed duration to capture the complete payload maneuver.
    normalize_observations=True,   # Normalize observations for stable training.
    action_repeat=1,               # Use high-frequency control (one action per timestep) for agile quadrotor behavior.
    unroll_length=10,              # Collect sequences of 10 timesteps per rollout to capture short-term dynamics.
    num_minibatches=32,            # Split the full batch into 32 minibatches to help stabilize the gradient updates.
    num_updates_per_batch=4,       # Apply 4 SGD updates per batch of data.
    discounting=0.97,              # Standard discount factor to balance immediate and future rewards.
    learning_rate=3e-4,            # A common starting learning rate that works well in many Brax tasks.
    entropy_cost=1e-2,             # Encourage exploration with a modest entropy bonus.
    num_envs=2048,                 # Run 2048 parallel environment instances for efficient data collection.
    batch_size=1024,               # Use a batch size that balances throughput with memory usage.
    seed=1                        # A fixed seed for reproducibility.
)

x_data, y_data, ydataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])
  plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')
  plt.errorbar(x_data, y_data, yerr=ydataerr)
  plt.savefig('mjx_brax_multiquad_policy.png')
  print(f'it/s: {num_steps / (times[-1] - times[0]).total_seconds()}')

  

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

# Save the trained policy.
model_path = '/tmp/mjx_brax_multiquad_policy'
model.save_params(model_path, params)

# Load parameters and define the inference function.
params = model.load_params(model_path)
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

eval_env = envs.get_environment(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]
n_steps = 500
render_every = 2
for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)
  if state.done:
    break

# media.show_video(
#     eval_env.render(rollout[::render_every]),
#     fps=1.0 / eval_env.dt / render_every
# )

# ----------------------------------------
# Demonstrate MJX Policy in MuJoCo (Optional)
# ----------------------------------------
mj_model = eval_env.sys.mj_model
mj_data = mj_model.make_data()

renderer = mjcf.Renderer(mj_model)
ctrl = jp.zeros(mj_model.nu)

images = []
for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  # For rendering, extract last_action from state.obs.
  obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data),
                          state.obs[-(eval_env.sys.nu+1):-1])
  ctrl, _ = jit_inference_fn(obs, act_rng)
  mj_data.ctrl = ctrl
  for _ in range(eval_env._n_frames):
    mjcf.mj_step(mj_model, mj_data)
  if i % render_every == 0:
    renderer.update_scene(mj_data)
    images.append(renderer.render())

# media.show_video(images, fps=1.0 / eval_env.dt / render_every)