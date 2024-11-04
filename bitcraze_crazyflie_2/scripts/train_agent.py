import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    env_id = 'DroneEnv-v0'

    num_envs = 64  # Adjusted number of environments
    n_steps = 512  # Increased n_steps
    total_timesteps_per_update = num_envs * n_steps  # 32 * 1024 = 32,768
    batch_size = 256  # Should be a factor of total_timesteps_per_update

    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv, monitor_dir="./logs")
    eval_env = make_vec_env(env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=0)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')

    # Adjust checkpoint and eval frequencies
    checkpoint_callback = CheckpointCallback(
        save_freq=5 * total_timesteps_per_update,  # Save every 5 updates
        save_path=models_dir,
        name_prefix='ppo_drone'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=models_dir,
        eval_freq=n_steps,  # Evaluate every update
        deterministic=True,
        render=False
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        device='auto',
        policy_kwargs=policy_kwargs
    )

    time_steps = 10_000_000
    model.learn(
        total_timesteps=time_steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save(model_path)
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
