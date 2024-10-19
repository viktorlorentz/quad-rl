import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import bitcraze_crazyflie_2  # Ensure your environment is registered

def main():
    # Create and wrap the training environment
    env = gym.make('DroneEnv-v0')
    env = Monitor(env)

    # Path to save models and logs
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ppo_drone')
    
    #clear previous checkpoint models
    for file in os.listdir(models_dir):
        if file.startswith('ppo_drone_'):
            os.remove(os.path.join(models_dir, file))



    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=models_dir,
                                             name_prefix='ppo_drone')

    # Create and wrap the evaluation environment
    eval_env = gym.make('DroneEnv-v0')
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                 log_path=models_dir, eval_freq=30000,
                                 deterministic=True, render=False)

    # Initialize the model
    model = PPO('MlpPolicy', env, tensorboard_log=models_dir)

    # Train the model
    time_steps = 10000000  # Adjust as needed
    model.learn(total_timesteps=time_steps, callback=[checkpoint_callback,eval_callback], progress_bar=True)

    # Save the final model
    model.save(model_path)

    # Close the environments
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
