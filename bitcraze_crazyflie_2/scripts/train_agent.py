import os
import bitcraze_crazyflie_2.envs.drone_env
import gymnasium as gym
import torch
import wandb
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback


# Define the custom callback for logging reward components
class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging reward components to wandb.
    """

    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # infos is a list of dicts, one for each environment
        infos = self.locals.get("infos", [])

        # Aggregate reward components across all environments
        reward_components = {}
        for info in infos:
            if "reward_components" in info:
                for key, value in info["reward_components"].items():
                    if key not in reward_components:
                        reward_components["reward/" + key] = []
                    reward_components["reward/" + key].append(value)

        # Compute mean of each component across environments
        if reward_components:
            mean_reward_components = {
                key: sum(values) / len(values)
                for key, values in reward_components.items()
            }

            # Log to wandb
            wandb.log(mean_reward_components, commit=False)

        return True


def main():

    env_id = "DroneEnv-v0"

    # Define parameters
    num_envs = 16
    n_steps = 512
    batch_size = 512
    time_steps = 3_000_000

    # Reward function coefficients
    reward_coefficients = {
        "distance_z": 1.0,
        "distance_xy": 1.0,
        "rotation_penalty": 2.0,
        "z_angular_velocity": 0.2,
        "angular_velocity": 0.01,
        "collision_penalty": 10.0,
        "terminate_collision": True,
        "out_of_bounds_penalty": 10.0,
        "alive_reward": 1.0,
        "linear_velocity": 0.5,
        "goal_bonus": 20.0,
        "distance": 0.0,
    }

    # Config for wandb
    default_config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": time_steps,
        "env_name": env_id,
        "n_steps": n_steps,
        "num_envs": num_envs,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "policy_kwargs": {
            "activation_fn": "Tanh",
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
        "reward_coefficients": reward_coefficients,
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
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "reward_coefficients": config["reward_coefficients"],
            "render_mode": None,
        },
        monitor_dir=f"monitor/{run.id}",
    )

    # Create the evaluation environment

    def trigger(t):
        if t % 50 == 0:
            # save video to global variable

            return True
        if t % 50 == 1:
            video = f"videos/{run.id}/rl-video-episode-{t-1}.mp4"
            run.log({"videos": wandb.Video(video)})

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
