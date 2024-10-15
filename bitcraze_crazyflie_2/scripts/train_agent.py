import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import bitcraze_crazyflie_2  # Ensure your environment is registered

def main():
    # Create the environment
    env = gym.make('DroneEnv-v0')

    # Path to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir,
                                             name_prefix='ppo_drone')
    eval_env = gym.make('DroneEnv-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                 log_path=models_dir, eval_freq=5000,
                                 deterministic=True, render=False)

    # Initialize the model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=models_dir)

    # Train the model
    time_steps = 100000  # Adjust as needed
    model.learn(total_timesteps=time_steps, callback=[checkpoint_callback, eval_callback])

    # Save the final model
    model.save(model_path)

    # Close the environments
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()