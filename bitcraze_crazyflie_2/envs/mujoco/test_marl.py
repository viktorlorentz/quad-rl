from datetime import datetime
import functools
import os
import time
import io

from typing import Any, Dict

import jax
from jax import numpy as jp
import numpy as np
from ml_collections import config_dict
import wandb
import imageio
import matplotlib.pyplot as plt

# JAX/Flax/Brax imports (as in your original file)
from flax.training import orbax_utils
from flax import struct
import mujoco
from mujoco import mjx
from brax import envs, base, math
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import mjcf, model

# For rendering and plotting
import mediapy as media
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# Import JAXMarl’s MAPPO algorithm (assumed API – please adjust per your version)
import jaxmarl.algorithms.mappo as mappo

# Set JAX platform to GPU if available.
jax.config.update('jax_platform_name', 'gpu')

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def lr_schedule(step, avg_episode_length, base_lr=3e-4):
    threshold = 1000  # target episode length in timesteps
    factor = 0.001 if avg_episode_length > threshold else 1.0
    return base_lr * factor

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

# --------------------------------------------------
# Original single-agent MultiQuadEnv (unchanged)
# --------------------------------------------------
class MultiQuadEnv(PipelineEnv):
    """
    A Brax/MJX version of a multi-rotor quadcopter team with payload.
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

        self.policy_freq = policy_freq
        self.sim_steps_per_action = sim_steps_per_action
        self.time_per_action = 1.0 / self.policy_freq
        self.max_time = max_time
        self._reset_noise_scale = reset_noise_scale
        self.warmup_time = 1.0

        dt = self.time_per_action / self.sim_steps_per_action
        sys.mj_model.opt.timestep = dt

        self.max_thrust = 0.11772
        self.goal_center = jp.array([0.0, 0.0, 1])
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

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        data = self.pipeline_init(qpos, qvel)
        last_action = jp.zeros(self.sys.nu)
        metrics = {'time': data.time, 'reward': jp.array(0.0)}
        obs = self._get_obs(data, last_action, self.target_position)
        reward = jp.array(0.0)
        done = jp.array(0.0)
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        prev_last_action = state.obs[-(self.sys.nu+1):-1]
        thrust_cmds = 0.5 * (action + 1.0)
        action_scaled = thrust_cmds * self.max_thrust
        data = self.pipeline_step(state.pipeline_state, action_scaled)
        target = self.target_position
        q1_orientation = data.xquat[self.q1_body_id]
        q2_orientation = data.xquat[self.q2_body_id]
        up = jp.array([0.0, 0.0, 1.0])
        q1_local_up = jp_R_from_quat(q1_orientation)[:, 2]
        q2_local_up = jp_R_from_quat(q2_orientation)[:, 2]
        angle_q1 = jp_angle_between(q1_local_up, up)
        angle_q2 = jp_angle_between(q2_local_up, up)
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
        out_of_bounds = jp.logical_or(out_of_bounds, data.xpos[self.payload_body_id][2] < 0.05)
        obs = self._get_obs(data, prev_last_action, target)
        reward, _, _ = self.calc_reward(
            obs, data.time, collision, out_of_bounds, action_scaled,
            angle_q1, angle_q2, prev_last_action, target
        )
        done = jp.logical_or(out_of_bounds, collision)
        done = jp.logical_or(done, data.time > self.max_time)
        done = done * 1.0
        new_metrics = {'time': data.time, 'reward': reward}
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=new_metrics)

    def _get_obs(self, data, last_action: jp.ndarray, target_position) -> jp.ndarray:
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
        quad2_linear_acc = data.cacc[self.q2_body_id][3:6]
        quad2_angular_acc = data.cacc[self.q2_body_id][:3]
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
        team_obs = obs[:6]
        quad1_obs = obs[6:30]
        quad2_obs = obs[30:54]
        quad_distance = jp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])
        payload_error = team_obs[:3]
        payload_linvel = team_obs[3:6]
        linvel_penalty = jp.linalg.norm(payload_linvel)
        dis = jp.linalg.norm(payload_error)
        z_error = jp.abs(payload_error[2])
        distance_reward = (1 - dis + jp.exp(-10 * dis)) + jp.exp(-10 * z_error)
        norm_error = jp.maximum(jp.linalg.norm(payload_error), 1e-6)
        norm_linvel = jp.maximum(jp.linalg.norm(payload_linvel), 1e-6)
        velocity_towards_target = jp.dot(payload_error, payload_linvel) / (norm_error * norm_linvel)
        safe_distance_reward = jp.clip((quad_distance - 0.11) / (0.15 - 0.11), 0, 1)
        collision_penalty = 5.0 * collision
        out_of_bounds_penalty = 50.0 * out_of_bounds
        smooth_action_penalty = jp.mean(jp.abs(action - last_action) / self.max_thrust)
        action_energy_penalty = jp.mean(jp.abs(action)) / self.max_thrust
        quad1_rel = quad1_obs[:3]
        quad2_rel = quad2_obs[:3]
        z_reward_q1 = quad1_rel[2] - target_position[2]
        z_reward_q2 = quad2_rel[2] - target_position[2]
        quad_above_reward = z_reward_q1 + z_reward_q2
        up_reward = jp.exp(-jp.abs(angle_q1)) + jp.exp(-jp.abs(angle_q2))
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

# Optionally register the single-agent environment.
envs.register_environment('multiquad', MultiQuadEnv)

# --------------------------------------------------
# New Multi-Agent Environment for JAXMarl
# --------------------------------------------------
class MultiAgentMultiQuadEnv(MultiQuadEnv):
    """
    A multi-agent version of the MultiQuadEnv.
    Each quad (agent) receives its own observation (shared payload info + individual quad state).
    The step() function takes a dictionary of actions from each agent,
    combines them, and returns a dictionary of per-agent states.
    """
    def reset(self, rng: jp.ndarray) -> Dict[str, Any]:
        state = super().reset(rng)
        shared = state.obs[:6]
        obs_agent0 = state.obs[6:30]
        obs_agent1 = state.obs[30:54]
        state_agent0 = state.replace(obs=jp.concatenate([shared, obs_agent0]))
        state_agent1 = state.replace(obs=jp.concatenate([shared, obs_agent1]))
        return {"agent_0": state_agent0, "agent_1": state_agent1}

    def step(self, state_dict: Dict[str, Any], actions: Dict[str, jp.ndarray]) -> Dict[str, Any]:
        joint_action = jp.concatenate([actions["agent_0"], actions["agent_1"]])
        state = state_dict["agent_0"]
        joint_state = super().step(state, joint_action)
        shared = joint_state.obs[:6]
        obs_agent0 = joint_state.obs[6:30]
        obs_agent1 = joint_state.obs[30:54]
        reward_agent0 = joint_state.reward
        reward_agent1 = joint_state.reward
        done = joint_state.done
        state_agent0 = joint_state.replace(obs=jp.concatenate([shared, obs_agent0]),
                                             reward=reward_agent0, done=done)
        state_agent1 = joint_state.replace(obs=jp.concatenate([shared, obs_agent1]),
                                             reward=reward_agent1, done=done)
        return {"agent_0": state_agent0, "agent_1": state_agent1}

# --------------------------------------------------
# Instantiate multi-agent environment and prepare JIT functions.
# --------------------------------------------------
env = MultiAgentMultiQuadEnv(policy_freq=250, sim_steps_per_action=1, max_time=10.0, reset_noise_scale=0.2)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# --------------------------------------------------
# JAXMarl Decentralized MAPPO Training Setup
# --------------------------------------------------
# Here we adjust the network architecture:
# - Agent policies: hidden layers of size (128, 64, 64)
# - Privileged critic (value network): hidden layers of size (128, 128, 128, 128)
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(128, 64, 64),
    value_hidden_layer_sizes=(128, 128, 128, 128)
)
agent_networks = {
    "agent_0": make_networks_factory(),
    "agent_1": make_networks_factory(),
}

# Define training configuration.
train_config = {
    "num_timesteps": 250_000_000,
    "num_evals": 10,
    "episode_length": 2000,
    "discounting": 0.99,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "num_envs": 2048,
    "batch_size": 256,
    "seed": 1,
    "unroll_length": 40,
    "num_minibatches": 8,
    "num_updates_per_batch": 4,
}

wandb.init(project="single_quad_rl", name=f"quad_marl_{int(time.time())}")

def progress(num_steps, metrics):
    print(f"Time: {datetime.now()}, Step: {num_steps}, Metrics: {metrics}")
    wandb.log({"num_steps": num_steps, **metrics})

# Train decentralized MAPPO using JAXMarl.
policies, training_info = mappo.train(
    env=env,
    agent_networks=agent_networks,
    config=train_config,
    progress_fn=progress
)

# Save the trained policies.
model_path = '/tmp/mappo_multi_quad_policy'
model.save_params(model_path, policies)

# Load policies (for inference).
policies = model.load_params(model_path)

# Create inference functions for each agent.
inference_fn_agent0 = jax.jit(policies["agent_0"].inference_fn)
inference_fn_agent1 = jax.jit(policies["agent_1"].inference_fn)

# --------------------------------------------------
# Inference and Evaluation Loop
# --------------------------------------------------
eval_env = MultiAgentMultiQuadEnv(policy_freq=250, sim_steps_per_action=1, max_time=10.0, reset_noise_scale=0.2)
jit_reset_eval = jax.jit(eval_env.reset)
jit_step_eval = jax.jit(eval_env.step)

rng = jax.random.PRNGKey(0)
state_dict = jit_reset_eval(rng)
rollout = [state_dict["agent_0"].pipeline_state]

n_steps = 2500
for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    action0, _ = inference_fn_agent0(state_dict["agent_0"].obs, act_rng)
    action1, _ = inference_fn_agent1(state_dict["agent_1"].obs, act_rng)
    actions = {"agent_0": action0, "agent_1": action1}
    state_dict = jit_step_eval(state_dict, actions)
    rollout.append(state_dict["agent_0"].pipeline_state)

# --------------------------------------------------
# Video Rendering
# --------------------------------------------------
def save_video(frames, filename, fps=30):
    try:
        imageio.mimsave(filename, frames, fps=float(fps))
        print(f"Video saved to {filename}")
    except ImportError:
        print("Could not save video. Install OpenCV or imageio.")

ctx = mujoco.GLContext(1920, 1080)
ctx.make_current()
frames = eval_env.render(rollout[::2], camera='track', width=1920, height=1080)
video_filename = "trained_policy_video.mp4"
save_video(frames, video_filename, fps=1.0 / eval_env.dt / 2)
wandb.log({"trained_policy_video": wandb.Video(video_filename, format="mp4")})

# --------------------------------------------------
# Example Plot: Histogram of Quad Actions (Placeholder)
# --------------------------------------------------
plt.figure()
plt.hist(np.random.randn(1000), bins=50)
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quad Actions')
plt.savefig('quad_actions_histogram.png')
print("Plot saved: quad_actions_histogram.png")
wandb.log({"quad_actions_histogram": wandb.Image('quad_actions_histogram.png')})
plt.close()

# --------------------------------------------------
# Example Plot: 3D Payload Trajectory
# --------------------------------------------------
payload_id = eval_env.payload_body_id
payload_positions = [np.array(s.xpos[payload_id]) for s in rollout]
payload_positions = np.stack(payload_positions)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(payload_positions[:, 0], payload_positions[:, 1], payload_positions[:, 2],
        label='Payload Trajectory', lw=2)
goal = np.array(eval_env.target_position)
ax.scatter(goal[0], goal[1], goal[2], color='red', s=50, label='Goal Position')
start_pos = payload_positions[0]
ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=50, label='Start Position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Payload Trajectory')
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.zaxis.set_major_locator(MaxNLocator(5))
ax.set_zlim(0, 1.5)
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300)
buf.seek(0)
img = Image.open(buf)
wandb.log({"payload_trajectory": wandb.Image(img)})
plt.close(fig)

print("Training, evaluation, and logging complete.")