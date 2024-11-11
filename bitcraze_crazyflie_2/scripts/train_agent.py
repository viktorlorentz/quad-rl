import os
import bitcraze_crazyflie_2.envs.drone_env
import gymnasium as gym
import torch
import wandb
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback


def main():

    env_id = "DroneEnv-v0"

    # Define parameters
    num_envs = 16  
    n_steps = 512  
    batch_size = 512  # Should be a factor of total_timesteps_per_update
    time_steps = 2_000_000  # Total training timesteps

    # Reward function coefficients
    reward_coefficients = {
        "distance_z": 1.0,
        "distance_xy": 1.0,
        "rotation_penalty": 2.0,
        "z_angular_velocity": 0.2,
        "angular_velocity": 0.01,
        "collision_penalty": 10.0,
        "out_of_bounds_penalty": 10.0,
        "alive_reward": 1.0,
        "linear_velocity": 0.1,
        "goal_bonus": 10.0,
        "distance" : 0
    }

    # Config for wandb (include important parameters for sweeps)
    config = {
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
        "reward_coefficients": reward_coefficients,  # Add reward coefficients to config
    }

    # Initialize wandb run
    run = wandb.init(
        project="single_quad_rl",  # Replace with your project name
        name=f"single_quad_rl_{int(time.time())}",
        config=config,
        sync_tensorboard=True,  # Auto-upload SB3's tensorboard metrics
        monitor_gym=True,  # Auto-upload videos of agent playing the game
        save_code=True,  # Optional
    )

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

    # # Define the environment creation function
    # def make_env(env_id, rank, seed=0, reward_coefficients=None):
    #     def _init():
    #         env = gym.make(env_id, reward_coefficients=reward_coefficients)
    #         env = Monitor(env)  # Record stats such as returns
    #         env.unwrapped.seed(seed + rank)
    #         return env
    #     return _init

    # # Create the vectorized environments
    # envs = [make_env('DroneEnv-v0', i, reward_coefficients=config["reward_coefficients"]) for i in range(config["num_envs"])]
    # env = SubprocVecEnv(envs)

    # env = gym.make_vec(
    #     env_id,
    #     num_envs=num_envs,
    #     vectorization_mode="async",
    #     reward_coefficients=config["reward_coefficients"],
    # )

    env = make_vec_env(
        env_id,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"reward_coefficients": config["reward_coefficients"], "render_mode":None},
        monitor_dir=f"monitor/{run.id}",
        
    )

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % config["n_steps"] * 100
    #     == 0,  # Record a video every 2000 steps
    #     video_length=2000,  # Length of recorded video
    #     name_prefix="rl-video",
    # )

    # # Wrap the environment with VecVideoRecorder to record videos
    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % config["n_steps"]*10 == 0,  # Record a video every 2000 steps
    #     video_length=200,  # Length of recorded video
    #     name_prefix="rl-video"
    # )

    # Create the evaluation environment
    # eval_env = gym.make_vec(
    #     env_id,
    #     num_envs=1,
    #     vectorization_mode="async",
    #     reward_coefficients=config["reward_coefficients"],
    # )

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

    # eval_env = gym.wrappers.RecordVideo(
    #     eval_env,
    #     video_folder=f"videos/{run.id}",
    #     episode_trigger=lambda x: x % 10 == 0,
    #     disable_logger=False,
    # )

    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     f"videos/{run.id}",
    #     record_video_trigger= lambda x: x % (config["n_steps"]*10) == 0,  # Record a video every 2000 steps
    #     video_length=2000,  # Length of recorded video
    #     name_prefix="rl-video"
    # )

    # eval_env =  gym.wrappers.RecordVideo(eval_env, video_folder= f"videos/{run.id}",
    #                                      step_trigger=lambda x: x % config["n_steps"]*10 == 0,
    #                                      disable_logger=True, video_length=200, )

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
        eval_freq=config["n_steps"] ,  # Evaluate every update
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

    # Start training with wandb callback
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            eval_callback,
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

    # delete video directory with files
    video_dir = f"videos/{run.id}"
    for file in os.listdir(video_dir):
        os.remove(os.path.join(video_dir, file))
    os.rmdir(video_dir)


if __name__ == "__main__":
    main()
