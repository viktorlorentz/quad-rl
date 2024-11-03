import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Function to create an environment instance
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)  # Attach a Monitor wrapper to record statistics
        env.unwrapped.seed(seed + rank)  # Ensure unique seeds for each instance
        return env
    return _init

def main():
    # Environment ID (Make sure your custom environment is registered)
    env_id = 'DroneEnv-v0'
    num_envs = 64  # Number of parallel environments

    # Create the vectorized environment (using SubprocVecEnv for multiprocessing)
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    
    # Eval Environment
    eval_env = SubprocVecEnv([make_env(env_id, 0)])

    # Directory to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')

    # Clear previous checkpoint models
    for file in os.listdir(models_dir):
        if file.startswith('ppo_drone_'):
            os.remove(os.path.join(models_dir, file))

    # Create callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=models_dir,
                                             name_prefix='ppo_drone')
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                 log_path=models_dir, eval_freq=100000,
                                 deterministic=True, render=False)

    # Initialize the PPO model with the vectorized environment
    model = PPO('MlpPolicy', env, tensorboard_log=models_dir,batch_size=128, device='cpu')
    # (policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)
    # Train the model with callbacks
    time_steps = 10000000  # Adjust as needed
    model.learn(total_timesteps=time_steps, callback=[checkpoint_callback, eval_callback], progress_bar=True)

    # Save the final model
    model.save(model_path)

    # Close the environments
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
