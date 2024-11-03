import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    # Environment ID (Make sure your custom environment is registered)
    env_id = 'DroneEnv-v0'
    num_envs = 96  # Number of parallel environments

    # Create the vectorized environment (using SubprocVecEnv for multiprocessing)
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv, monitor_dir="./logs")

    # Eval Environment
    eval_env = make_vec_env(env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=0)

    # Directory to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')

    # Clear previous checkpoint models
    for file in os.listdir(models_dir):
        if file.startswith('ppo_drone_'):
            os.remove(os.path.join(models_dir, file))

    # Create callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(save_freq=1000000/128, save_path=models_dir, name_prefix='ppo_drone')
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir, log_path=models_dir, 
                                 eval_freq=100000/128, deterministic=True, render=False)

    # Initialize the PPO model with the vectorized environment
    model = PPO('MlpPolicy', env, n_steps=2048, batch_size=512, device='cuda')

    # Train the model with callbacks
    time_steps = 10_000_000  # Adjust as needed
    model.learn(total_timesteps=time_steps, callback=[checkpoint_callback, eval_callback], progress_bar=True)

    # Save the final model
    model.save(model_path)

    # Close the environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
