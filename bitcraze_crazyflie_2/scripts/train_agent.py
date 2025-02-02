import os
from bitcraze_crazyflie_2.envs.drone_env import DroneEnv
import gymnasium as gym
import torch
import wandb
import time
import gc

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from wandb.integration.sb3 import WandbCallback


# Define the custom callback for logging reward components
class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging episode-average reward components and position tracking to wandb.
    Only uses the first environment for tracking.
    """

    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = {}
        self.episode_length = 0
        self.episode_distance_sum = 0.0  # To accumulate distance_to_target per episode
        self.actions_sum = 0
        self.actions_hist = np.zeros(20)

    def _on_step(self) -> bool:
        # Get locals
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # Only consider the first environment (index 0)
        info = infos[0]
        done = dones[0]

        # If 'reward_components' is in info, accumulate them
        if "reward_components" in info:
            for key, value in info["reward_components"].items():
                if key not in self.episode_rewards:
                    self.episode_rewards["reward/" + key] = 0.0
                self.episode_rewards["reward/" + key] += value

        # Accumulate distance_to_target
        if "distance_to_target" in info:
            self.episode_distance_sum += info["distance_to_target"]

        # Accumulate actions
        if "action" in info:
            self.actions_sum += sum(info["action"]) / 4
            self.actions_hist += np.histogram(
                info["action"], bins=20, range=(0, 0.118)
            )[0]

        if "env_randomness" in info:
            wandb.log({"env_randomness": info["env_randomness"]})
        
        if "average_episode_length" in info:
            wandb.log({"average_episode_length": info["average_episode_length"]})

        
        # Increment episode length
        self.episode_length += 1

        if len(self.episode_rewards) > 0 and done:
            # Compute average reward components for this episode
            additional_metrics = {
                key: value / self.episode_length
                for key, value in self.episode_rewards.items()
            }

            # Compute average position tracking
            position_tracking = self.episode_distance_sum / self.episode_length

            # this makes sure that it hovers first before tracking
            # any metric above 1 means its not hovering
            if self.episode_length < 3000:
                position_tracking = 1 + (3000 - self.episode_length) / 3000

            # Add position_tracking to the log data
            additional_metrics["position_tracking"] = position_tracking

            # Optionally, add mean episode length to the log
            additional_metrics["episode_length"] = self.episode_length

            # Log distribution of actions as histogram
            additional_metrics["actions/mean"] = self.actions_sum / self.episode_length

            # actions binned into 20
            self.actions_hist = self.actions_hist / np.sum(self.actions_hist)
            additional_metrics["actions/hist"] = wandb.Histogram(
                np_histogram=(self.actions_hist, np.linspace(0, 0.118, 21))
            )

            # action satuarion (actions that are in first and last bin)
            additional_metrics["actions/saturation"] = (
                self.actions_hist[0] + self.actions_hist[-1]
            )

            # Add saturation penalty to position tracking

            saturation_penalty = 2 * max(additional_metrics["actions/saturation"]-0.1, 0)

            additional_metrics["position_tracking"] += saturation_penalty

            # Log to wandb
            wandb.log(additional_metrics, commit=False)

            # Reset episode data
            self.episode_rewards = {}
            self.episode_length = 0
            self.episode_distance_sum = 0.0
            self.actions_sum = 0
            self.actions_hist = np.zeros(20)

        return True


class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Wrap the existing action_net with tanh activation
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


def main():

    env_id = "DroneEnv-v0"

    # Define parameters
    n_envs = 64
    n_steps = 2048
    batch_size = 64
    time_steps = 150_000_000

    # Reward function coefficients
    reward_coefficients = {  # based on single_quad_rl_1731931528
        "distance": 1,
        "distance_z": 0,
        "goal_bonus": 100,
        "distance_xy": 0,
        "alive_reward": 3,
        "linear_velocity": 0,
        "angular_velocity": 0,
        "rotation_penalty": 3,
        "collision_penalty": 0,
        "z_angular_velocity": 1,
        "terminate_collision": True,
        "out_of_bounds_penalty": 0,
        "velocity_towards_target": 0.5,
        "action_saturation": 0,
        "smooth_action": 0.5,
        "energy_penalty": 0.1,
        "payload_velocity": 0.05,
        "above_payload": 0.2,
    }

    # Config for wandb
    default_config = {  # based on single_quad_rl_1731931528
        "policy_type": "MlpPolicy",
        "total_timesteps": time_steps,
        "env_name": env_id,
        "n_steps": n_steps,
        "n_envs": n_envs,
        "batch_size": batch_size,
        "learning_rate": 0.0003,
        "gamma": 0.98,
        "gae_lambda": 0.83,
        "ent_coef": 0.05,
        "vf_coef": 0.25,
        "max_grad_norm": 0.5,
        "clip_range": 0.22,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "use_sde": False,
        "policy_kwargs": {
            "activation_fn": "Tanh",
            "net_arch": {"pi": [64, 64, 64], "vf": [64, 64, 64]},
            "squash_output": False,  # this adds tanh to the output of the policy
        },
        "reward_coefficients": reward_coefficients,
        "policy_freq": 250,
        "env_config": {
            "connect_payload": True,
            "randomness": 1.0,
            "target_mode": "payload",
            "curriculum" : True,
            "num_stack_frames": 3,
            "stack_stride": 1,
        }
    }

    # Initialize wandb run
    run = wandb.init(
        project="single_quad_rl",
        name=f"single_quad_rl_{int(time.time())}",
        config=default_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    config = wandb.config

    # Map activation function name from config to actual function
    activation_fn = getattr(torch.nn, config["policy_kwargs"]["activation_fn"])

    # Build policy_kwargs with correct net_arch
    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(
            pi=config["policy_kwargs"]["net_arch"]["pi"],
            vf=config["policy_kwargs"]["net_arch"]["vf"],
        ),
        squash_output=config["policy_kwargs"]["squash_output"],
    )

    # Create the vectorized environments
    env = make_vec_env(
        DroneEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "reward_coefficients": config["reward_coefficients"],
            "render_mode": None,
            "policy_freq": config["policy_freq"],
            "env_config": config["env_config"],
        },
        monitor_dir=f"monitor/{run.id}",
    )

    # Create the evaluation environment

    def trigger(t):
        if t % 150 == 0:
            # save video to global variable

            return True
        if t % 150 == 1:
            video = f"videos/{run.id}/rl-video-episode-{t-1}.mp4"
            run.log({"videos": wandb.Video(video)})
            gc.collect()

    eval_env = make_vec_env(
        DroneEnv,
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        seed=0,
        env_kwargs={
            "reward_coefficients": config["reward_coefficients"],
            "render_mode": "rgb_array",
            "env_config": config["env_config"],
            "policy_freq": config["policy_freq"],
        },
        wrapper_class=gym.wrappers.RecordVideo,
        wrapper_kwargs={
            "video_folder": f"videos/{run.id}",
            "episode_trigger": trigger,
        },
    )

    # Directory to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ppo_drone")

    # Clear previous models
    for file in os.listdir(models_dir):
        if file.startswith("ppo_drone_"):
            os.remove(os.path.join(models_dir, file))

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=models_dir,
        eval_freq=config["n_steps"],
        deterministic=True,
        render=False,
    )

    # Create the PPO model with all specified parameters
    model = PPO(
        policy='MlpPolicy', #config["policy_type"],  # CustomActorCriticPolicy,
        env=env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        # gamma=config["gamma"],
        # gae_lambda=config["gae_lambda"],
        # ent_coef=config["ent_coef"],
        # vf_coef=config["vf_coef"],
        # max_grad_norm=config["max_grad_norm"],
        # clip_range=config["clip_range"],
        # clip_range_vf=config["clip_range_vf"],
        # normalize_advantage=config["normalize_advantage"],
        # use_sde=config["use_sde"],
        device="cpu",
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run.id}",
    )

    # Start training with wandb and custom callbacks
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            eval_callback,
            RewardLoggingCallback(),
            WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        ],
        progress_bar=True,
    )

    # Save the final model
    model.save(model_path)
    env.close()
    eval_env.close()

    run.finish()

    # Optionally delete video directory with files
    video_dir = f"videos/{run.id}"
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)


if __name__ == "__main__":
    main()
