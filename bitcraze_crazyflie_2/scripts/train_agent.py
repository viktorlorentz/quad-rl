import os
import bitcraze_crazyflie_2.envs.drone_env
import gymnasium as gym
import torch
import wandb
import time
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
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

        # Increment episode length
        self.episode_length += 1

        if done and len(self.episode_rewards) > 0:
            # Compute average reward components for this episode
            avg_reward_components = {
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
            avg_reward_components["position_tracking"] = position_tracking

            # Optionally, add episode length to the log
            avg_reward_components["episode_length"] = self.episode_length

            # Log to wandb
            wandb.log(avg_reward_components, commit=False)

            # Reset episode data
            self.episode_rewards = {}
            self.episode_length = 0
            self.episode_distance_sum = 0.0

        return True


def main():

    env_id = "DroneEnv-v0"

    # Define parameters
    n_envs = 8
    n_steps = 1024
    batch_size = 128
    time_steps = 1_000_000

    # Reward function coefficients
    reward_coefficients = {  # based on single_quad_rl_1731931528
        "distance": 1,
        "distance_z": 0.5,
        "goal_bonus": 20,
        "distance_xy": 0.9,
        "alive_reward": 5,
        "linear_velocity": 0.6,
        "angular_velocity": 0.3,
        "rotation_penalty": 1,
        "collision_penalty": 200,
        "z_angular_velocity": 0.17,
        "terminate_collision": True,
        "out_of_bounds_penalty": 5,
        "velocity_towards_target": 5,
    }

    # Config for wandb
    default_config = {  # based on single_quad_rl_1731931528
        "policy_type": "MlpPolicy",
        "total_timesteps": time_steps,
        "env_name": env_id,
        "n_steps": n_steps,
        "n_envs": n_envs,
        "batch_size": batch_size,
        "learning_rate": 0.0012,
        "gamma": 0.98,
        "gae_lambda": 0.83,
        "ent_coef": 0.05,
        "vf_coef": 0.25,
        "max_grad_norm": 0.78,
        "clip_range": 0.22,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "policy_kwargs": {
            "activation_fn": "Tanh",
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        "reward_coefficients": reward_coefficients,
        "policy_freq": 200
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
    )

    # Create the vectorized environments
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "reward_coefficients": config["reward_coefficients"],
            "render_mode": None,
            "policy_freq": config["policy_freq"],
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
        env_id,
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        seed=0,
        env_kwargs={"reward_coefficients": config["reward_coefficients"]},
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
        policy=config["policy_type"],
        env=env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        normalize_advantage=config["normalize_advantage"],
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
