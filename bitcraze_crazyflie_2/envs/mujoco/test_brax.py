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

import wandb
import time
import imageio

jax.config.update('jax_platform_name', 'gpu')



def lr_schedule(step, avg_episode_length, base_lr=3e-4):
    threshold = 1000  # target episode length in timesteps
    factor = 0.001 if avg_episode_length > threshold else 1.0
    return base_lr * factor

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
      sim_steps_per_action: int = 1,           # Physics steps between control actions.
      max_time: float = 10.0,                  # Maximum simulation time per episode.
      reset_noise_scale: float = 0.1,          # Noise scale for initial state reset.
      **kwargs,
  ):
    # Load the MJX model from the XML file.
    mj_model = mujoco.MjModel.from_xml_path("two_quad_payload.xml")


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

    # set sim timestep based on freq and steps per action and set timestep
    dt = self.time_per_action / self.sim_steps_per_action
    sys.mj_model.opt.timestep = dt

    # Maximum thrust from original env.
    self.max_thrust = 0.11772
    # Replace target_position with a fixed goal center and add a sphere radius.
    self.goal_center = jp.array([0.0, 0.0, 1])
    self.goal_radius = 0.8  # sphere radius for random goal position
    self.target_position = self.goal_center
    
    # Cache body/geom IDs for faster lookup.
    self.payload_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.q1_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_cf2")
    self.q2_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_cf2")

    # Register the new goal marker body
    self.goal_site_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "goal_marker")

  
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    # Add random goal position in a sphere around the initial goal center.
    rng, rng_goal = jax.random.split(rng)
    # offset = jax.random.normal(rng_goal, shape=(3,))
    # offset = offset / jp.linalg.norm(offset) * (
    #     self.goal_radius * jax.random.uniform(rng_goal, shape=(), minval=0.0, maxval=1.0))
    # new_target = jax.lax.stop_gradient(self.goal_center + offset)

    # set site marker to target position 

 

    rng, rng1, rng2 = jax.random.split(rng, 3)
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
    data = self.pipeline_init(qpos, qvel)
  

    last_action = jp.zeros(self.sys.nu)

    metrics = {
        'time': data.time,
        'reward': jp.array(0.0),
    }
    obs = self._get_obs(data, last_action, self.target_position)
    reward = jp.array(0.0)
    done = jp.array(0.0)
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Advances the environment by one control step."""
    # Extract the previous last_action from the observation.
    prev_last_action = state.obs[-(self.sys.nu+1):-1]
    # Convert actions from [-1, 1] to thrust commands in [0, max_thrust].
    thrust_cmds = 0.5 * (action + 1.0)
    action_scaled = thrust_cmds * self.max_thrust

    data0 = state.pipeline_state

 

    data = self.pipeline_step(data0, action_scaled)


   
    target = self.target_position
    
    # Compute the tilt (angle from vertical) for each quad.
    q1_orientation = data.xquat[self.q1_body_id]
    q2_orientation = data.xquat[self.q2_body_id]
    up = jp.array([0.0, 0.0, 1.0])
    q1_local_up = jp_R_from_quat(q1_orientation)[:, 2]
    q2_local_up = jp_R_from_quat(q2_orientation)[:, 2]
    angle_q1 = jp_angle_between(q1_local_up, up)
    angle_q2 = jp_angle_between(q2_local_up, up)

    # Simple collision/out-of-bounds checks.
    quad1_pos = data.xpos[self.q1_body_id]
    quad2_pos = data.xpos[self.q2_body_id]
    quad_distance = jp.linalg.norm(quad1_pos - quad2_pos)
    collision = quad_distance < 0.11
    out_of_bounds = jp.logical_or(jp.absolute(angle_q1) > jp.radians(80),
                                  jp.absolute(angle_q2) > jp.radians(80))
    out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.q1_body_id][2] < 0.05)
    out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.q2_body_id][2] < 0.05)
    out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.q1_body_id][2] < data.xpos[self.payload_body_id][2])
    out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.q2_body_id][2] < data.xpos[self.payload_body_id][2])

    # out of bounds for payload
    out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.payload_body_id][2] < 0.05)

    obs = self._get_obs(data, prev_last_action, target)

    reward, _, _ = self.calc_reward(
        obs, data.time, collision, out_of_bounds, action_scaled,
        angle_q1, angle_q2, prev_last_action, target
    )

    # Terminate if collision, out-of-bounds, or max time reached.
    done = jp.logical_or(out_of_bounds, collision)
    done = jp.logical_or(done, data.time > self.max_time)
    done = done * 1.0

    new_metrics = {
        'time': data.time,
        'reward': reward
    }
    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=new_metrics)

  def _get_obs(self, data, last_action: jp.ndarray, target_position) -> jp.ndarray:
    """Constructs the observation vector from simulation data."""
    # Payload state.
    payload_pos = data.xpos[self.payload_body_id]
    payload_linvel = data.cvel[self.payload_body_id][3:6]
    payload_error = target_position - payload_pos

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
        last_action,          # (nu,)
    ])
    return obs

  def calc_reward(self, obs, sim_time, collision, out_of_bounds, action,
                  angle_q1, angle_q2, last_action, target_position):
    """
    Computes a reward similar to the original gym env by splitting the observation into team and quad parts.
    """
    tp = target_position
    # Team observation: payload error (3) and payload linear velocity (3).
    team_obs = obs[:6]
    # Quad observations: next 24 elements for quad1 and following 24 for quad2.
    quad1_obs = obs[6:30]
    quad2_obs = obs[30:54]
    quad_distance = jp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])

    payload_error = team_obs[:3]
    payload_linvel = team_obs[3:6]
    linvel_penalty = jp.linalg.norm(payload_linvel)
    dis = jp.linalg.norm(payload_error)
    # Emphasize z_error 
    z_error = jp.abs(payload_error[2])
    distance_reward = (1 - dis + jp.exp(-10 * dis)) + jp.exp(-10 * z_error)

    # Compute velocity alignment (dot product).
    norm_error = jp.maximum(jp.linalg.norm(payload_error), 1e-6)
    norm_linvel = jp.maximum(jp.linalg.norm(payload_linvel), 1e-6)
    velocity_towards_target = jp.dot(payload_error, payload_linvel) / (norm_error * norm_linvel)
  
    safe_distance_reward = jp.clip((quad_distance - 0.11) / (0.15 - 0.11), 0, 1)
    collision_penalty = 5.0 * collision
    out_of_bounds_penalty = 50.0 * out_of_bounds
    smooth_action_penalty = jp.mean(jp.abs(action - last_action) / self.max_thrust)
    action_energy_penalty = jp.mean(jp.abs(action)) / self.max_thrust
    
    # Reward for quad z position above the payload target.
    quad1_rel = quad1_obs[:3]
    quad2_rel = quad2_obs[:3]
    z_reward_q1 = quad1_rel[2] - tp[2]
    z_reward_q2 = quad2_rel[2] - tp[2]
    quad_above_reward = z_reward_q1 + z_reward_q2

    up_reward = jp.exp(-jp.abs(angle_q1)) + jp.exp(-jp.abs(angle_q2))

    # Penalties for angular and linear velocity of quads.
    ang_vel_q1 = quad1_obs[15:18]
    ang_vel_q2 = quad2_obs[15:18]
    ang_vel_penalty = 0.1 * (jp.linalg.norm(ang_vel_q1)**2 + jp.linalg.norm(ang_vel_q2)**2)
    linvel_q1 = quad1_obs[9:12]
    linvel_q2 = quad2_obs[9:12]
    linvel_quad_penalty = 0.1 * (jp.linalg.norm(linvel_q1)**2 + jp.linalg.norm(linvel_q2)**2)

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
  ctrl = 0.1 * jp.ones(env.sys.nu)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

# media.show_video(env.render(rollout), fps=1.0 / env.dt)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 128, 128, 128)
)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=200_000_000,
    num_evals=10,
    reward_scaling=1,
    episode_length=2000,
    normalize_observations=False,
    action_repeat=1,
    unroll_length=40,
    num_minibatches=8,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=256,
    seed=1,
    network_factory=make_networks_factory
)

x_data, y_data, ydataerr = [], [], []
times = [datetime.now()]

wandb.init(project="single_quad_rl", name=f"quad_rl_{int(time.time())}")

def save_video(frames, filename, fps=30):
  try:
      imageio.mimsave(filename, frames, fps=float(fps))
      print(f"Video saved to {filename}")
  except ImportError:
      print("Could not save video. Install OpenCV or imageio.")

def render_video(video_filename, env, duration=5.0, framerate=30):
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=1920, height=1080)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    while mj_data.time < duration:
        mujoco.mj_step(mj_model, mj_data)
        if len(frames) < mj_data.time * framerate:
            renderer.update_scene(mj_data, camera="track", scene_option=scene_option)
            frame = renderer.render()
            frames.append(frame)
    renderer.close()
    save_video(frames, video_filename, fps=framerate)

def progress(num_steps, metrics):

    times.append(datetime.now())

  

    it_per_sec = num_steps / (times[-1] - times[0]).total_seconds()
    progress_val = num_steps / train_fn.keywords['num_timesteps']
    reward =  metrics['eval/episode_reward']
    elapsed_time = times[-1] - times[0]
    print(f'time: {elapsed_time}, step: {num_steps}, progress: {progress_val:.1%}, reward: {reward:.3f}, it/s: {it_per_sec:.1f}')
    wandb.log({"num_steps": num_steps, **metrics})

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

model_path = '/tmp/mjx_brax_multiquad_policy'
model.save_params(model_path, params)

params = model.load_params(model_path)
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

eval_env = envs.get_environment(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]


# --------------------
# Video Rendering 
# --------------------
n_steps = 2500
render_every = 2
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# Initialize list to record quad actions.
quad_actions_list = []

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)
    # Record the actions.
    quad_actions_list.append(np.array(ctrl))
    
ctx = mujoco.GLContext(1920, 1080)
ctx.make_current()

frames = eval_env.render(rollout[::render_every], camera='track', width=1920, height=1080)
video_filename = "trained_policy_video.mp4"
save_video(frames, video_filename, fps=1.0 / eval_env.dt / render_every)
wandb.log({"trained_policy_video": wandb.Video(video_filename, format="mp4")})

# Histogram plot over quad actions.
quad_actions_flat = np.concatenate(quad_actions_list).flatten()
plt.figure()
plt.hist(quad_actions_flat, bins=50)
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quad Actions')
plt.savefig('quad_actions_histogram.png')
print("Plot saved: quad_actions_histogram.png")
wandb.log({"quad_actions_histogram": wandb.Image('quad_actions_histogram.png')})
plt.close()

# --------------------
# 3D Trajectory Plot for Payload
# --------------------
import io
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image  # add this import
import matplotlib.ticker as mticker

# Extract payload positions from rollout
# Note: rollout elements are pipeline_state objects with xpos attribute.
payload_id = eval_env.payload_body_id
payload_positions = [np.array(s.xpos[payload_id]) for s in rollout]
payload_positions = np.stack(payload_positions)  # shape: (T, 3)

fig = plt.figure( figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(payload_positions[:,0], payload_positions[:,1], payload_positions[:,2],
        label='Payload Trajectory', lw=2)
# Mark the goal position with a red dot
goal = np.array(eval_env.target_position)
ax.scatter(goal[0], goal[1], goal[2], color='red', s=50, label='Goal Position')
# Mark the start position with a green dot
start_pos = payload_positions[0]
ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=50, label='Start Position')

# Extract trajectories for Quads
quad1_positions = [np.array(s.xpos[eval_env.q1_body_id]) for s in rollout]
quad2_positions = [np.array(s.xpos[eval_env.q2_body_id]) for s in rollout]
quad1_positions = np.stack(quad1_positions)  # shape: (T, 3)
quad2_positions = np.stack(quad2_positions)  # shape: (T, 3)

# Plot dashed trajectories for quads in different colors
ax.plot(quad1_positions[:,0], quad1_positions[:,1], quad1_positions[:,2],
  ls='--', color='blue', lw=2, alpha=0.5, label='Quad1 Trajectory')
ax.plot(quad2_positions[:,0], quad2_positions[:,1], quad2_positions[:,2],
  ls='--', color='magenta', lw=2, alpha=0.5, label='Quad2 Trajectory')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Payload Trajectory')

# Use fewer axis ticks
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
ax.zaxis.set_major_locator(mticker.MaxNLocator(5))

ax.set_zlim(0, 1.5)  

# Save the plot to a bytes buffer with higher dpi
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300)
print("Plot saved: 3D payload trajectory plot")
buf.seek(0)
img = Image.open(buf)  # convert buffer to PIL image
wandb.log({"payload_trajectory": wandb.Image(img)})
plt.close(fig)

# --------------------
# Top-Down (XY) Trajectory Plot for Payload
# --------------------
fig_topdown = plt.figure(figsize=(5, 5))
plt.plot(payload_positions[:,0], payload_positions[:,1],
         label='Payload XY Trajectory', lw=2)
# Add quad trajectories (XY)
plt.plot(quad1_positions[:,0], quad1_positions[:,1],
         ls='--', color='blue', lw=2, alpha=0.7, label='Quad1 XY Trajectory')
plt.plot(quad2_positions[:,0], quad2_positions[:,1],
         ls='--', color='magenta', lw=2, alpha=0.7, label='Quad2 XY Trajectory')
plt.scatter(goal[0], goal[1], color='red', s=50, label='Goal Position')
plt.scatter(start_pos[0], start_pos[1], color='green', s=50, label='Start Position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Payload Trajectory (Top Down)')
plt.legend()
buf_top = io.BytesIO()
plt.savefig(buf_top, format='png', dpi=300)
print("Plot saved: Top-down payload trajectory plot")
buf_top.seek(0)
img_top = Image.open(buf_top)
wandb.log({"payload_trajectory_topdown": wandb.Image(img_top)})
plt.close(fig_topdown)

# --------------------
# Payload Position Error Over Time Plot
# --------------------

times_sim = np.array([s.time for s in rollout])
payload_errors = np.array([
    np.linalg.norm(np.array(s.xpos[eval_env.payload_body_id]) - np.array(eval_env.target_position))
    for s in rollout
])
fig2 = plt.figure()
plt.plot(times_sim, payload_errors, linestyle='-', color='orange', label='Payload Error')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Payload Position Error')
plt.title('Payload Position Error Over Time')
plt.legend()
buf2 = io.BytesIO()
plt.savefig(buf2, format='png', dpi=300)
print("Plot saved: Payload error over time plot")
buf2.seek(0)

img2 = Image.open(buf2)
wandb.log({"payload_error_over_time": wandb.Image(img2)})
plt.close(fig2)

# --------------------
# Batched Rollout over 100 Envs and Top-Down XY Plot for Final Positions 

num_envs = 100
n_steps = 2500

# Create 100 independent environment instances.
batched_rngs = jax.random.split(jax.random.PRNGKey(1234), num_envs)
batched_states = jax.vmap(jit_reset)(batched_rngs)

# Record starting payload positions (XY) from each environment.
start_positions = jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.payload_body_id])(batched_states)
start_positions = np.array(start_positions)  # shape: (num_envs, 3)

# --- Begin batched rollout with error recording ---
batched_errors = []  # Will store per-env payload errors for every timestep
timeline = []        # Will store simulation time for each timestep

rng_main = jax.random.PRNGKey(5678)
for step in range(n_steps):
    rng_main, rng_step = jax.random.split(rng_main)
    act_rngs = jax.random.split(rng_step, num_envs)
    ctrls, _ = jax.vmap(jit_inference_fn)(batched_states.obs, act_rngs)
    batched_states = jax.vmap(jit_step)(batched_states, ctrls)

    # Record payload error for each env.
    errors = jax.vmap(lambda s: jax.numpy.linalg.norm(s.pipeline_state.xpos[eval_env.payload_body_id] - eval_env.target_position))(batched_states)
    batched_errors.append(np.array(errors))
    # Record simulation time (all envs have the same time).
    times_env = jax.vmap(lambda s: s.pipeline_state.time)(batched_states)
    timeline.append(np.array(times_env[0]))
# --- End batched rollout with error recording ---

# Extract final positions for payload, quad1, and quad2 (only XY coordinates).
final_payload_positions = jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.payload_body_id])(batched_states)
final_quad1_positions   = jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.q1_body_id])(batched_states)
final_quad2_positions   = jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.q2_body_id])(batched_states)

final_payload_positions = np.array(final_payload_positions)[:, :2]
final_quad1_positions   = np.array(final_quad1_positions)[:, :2]
final_quad2_positions   = np.array(final_quad2_positions)[:, :2]

fig, ax = plt.subplots(figsize=(8, 8))
# Mark start payload positions.
ax.scatter(start_positions[:, 0], start_positions[:, 1],
           color='black', s=10, label='Start Payload')
# Mark the goal position.
goal = np.array(eval_env.target_position)
ax.scatter(goal[0], goal[1], color='red', s=70, marker='*', label='Goal Position')

# --- Begin modifications ---
from matplotlib.colors import LinearSegmentedColormap
hot = plt.cm.get_cmap('hot', 256)
newcolors = hot(np.linspace(0, 1, 256))
newcolors[0, :] = np.array([1, 1, 1, 0])  # Set lowest value to transparent white
new_cmap = LinearSegmentedColormap.from_list('hot_modified', newcolors)

# Remove heatmap and add contour plot with fixed limits.
all_x = final_payload_positions[:, 0]
all_y = final_payload_positions[:, 1]
# Set axis limits to -0.3 and 0.3.
x_low, x_high = -0.3, 0.3
y_low, y_high = -0.3, 0.3
# Create mask for inliers.
inliers_mask = ((final_payload_positions[:, 0] >= x_low) &
                (final_payload_positions[:, 0] <= x_high) &
                (final_payload_positions[:, 1] >= y_low) &
                (final_payload_positions[:, 1] <= y_high))
inliers = final_payload_positions[inliers_mask]
outliers = final_payload_positions[~inliers_mask]
# Compute 2D histogram for inliers.
xbins = np.linspace(x_low, x_high, 30)
ybins = np.linspace(y_low, y_high, 30)
H, xedges, yedges = np.histogram2d(inliers[:, 0], inliers[:, 1], bins=[xbins, ybins], density=True)
# Compute grid for contour plot.
Xc = (xedges[:-1] + xedges[1:]) / 2
Yc = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(Xc, Yc)
# Plot the contour using the new colormap and force 0 to be white/transparent.
ax.contourf(X, Y, H.T, levels=10, cmap=new_cmap, alpha=0.7, vmin=0)
# Plot outliers separately.
if outliers.size > 0:
    ax.scatter(outliers[:, 0], outliers[:, 1], color='cyan', marker='x', s=20, label='Outliers')
# Set axis limits.
ax.set_xlim(x_low, x_high)
ax.set_ylim(y_low, y_high)
# --- End modifications ---

# Mark final quad positions as small squares with low opacity.
ax.scatter(final_quad1_positions[:, 0], final_quad1_positions[:, 1],
           color='blue', marker='s', s=15, alpha=0.3, label='Quad1 Final')
ax.scatter(final_quad2_positions[:, 0], final_quad2_positions[:, 1],
           color='magenta', marker='s', s=15, alpha=0.3, label='Quad2 Final')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Top-Down XY Plot for Final Positions (Batched Rollout)')
ax.legend()

buf_final = io.BytesIO()
plt.savefig(buf_final, format='png', dpi=300)
buf_final.seek(0)
img_final = Image.open(buf_final)
wandb.log({"batched_rollout_topdown": wandb.Image(img_final)})
print("Plot saved and logged: batched_rollout_topdown")
plt.close(fig)

# --------------------
# Batched Payload Error Over Time Plot using percentiles
timeline = np.array(timeline)  # shape: (n_steps,)
batched_errors = np.array(batched_errors)  # shape: (n_steps, num_envs)

p0 = np.percentile(batched_errors, 0, axis=1)
p25 = np.percentile(batched_errors, 25, axis=1)
p50 = np.percentile(batched_errors, 50, axis=1)
p75 = np.percentile(batched_errors, 75, axis=1)
p95 = np.percentile(batched_errors, 95, axis=1)

fig3 = plt.figure(figsize=(8, 5))
ax3 = fig3.add_subplot(111)
ax3.plot(timeline, p0, color='black', linestyle='--', label='0th Percentile')
ax3.plot(timeline, p25, color='blue', linestyle='-.', label='25th Percentile')
ax3.plot(timeline, p50, color='blue', linewidth=2, label='50th Percentile')
ax3.plot(timeline, p75, color='blue', linestyle='-.', label='75th Percentile')
ax3.plot(timeline, p95, color='black', linestyle='--', label='95th Percentile')

ax3.set_xlabel('Simulation Time (s)')
ax3.set_ylabel('Payload Position Error')
ax3.set_title('Batched Rollout Payload Position Error Over Time')
ax3.legend()
ax3.grid(True)
plt.savefig('batched_payload_error_over_time.png', dpi=300)
print("Plot saved: Batched Payload Error Over Time")
wandb.log({"batched_payload_error_over_time": wandb.Image('batched_payload_error_over_time.png')})
plt.close(fig3)
