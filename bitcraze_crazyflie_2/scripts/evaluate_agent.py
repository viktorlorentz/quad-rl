import os
import gymnasium as gym
from stable_baselines3 import PPO
import bitcraze_crazyflie_2  # Ensure your environment is registered

def main():
    # Create the environment with rendering enabled
    env = gym.make('DroneEnv-v0', render_mode='human')

    # Path to the saved model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(models_dir, 'ppo_drone.zip')

    # Load the trained model
    model = PPO.load(model_path)

    # Evaluate the model
    obs, info = env.reset()
    terminated = False
    truncated = False

    last_reward = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        last_reward = reward

    print(f'Final reward: {last_reward}')

    env.close()

if __name__ == '__main__':
        main()
