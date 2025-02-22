import os
import gymnasium as gym
from stable_baselines3 import PPO
import bitcraze_crazyflie_2  # Ensure your environment is registered

def main():
    # Create the environment with rendering enabled
    env = gym.make(
        'MultiQuadEnv',
        render_mode='human',
        env_config={
            "randomness": 1.0, 
            "connect_payload": False,
            "max_time": 60,
            "velocity_observaiton": True,
        }
    )

    # Path to the saved model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(models_dir, 'best_model.zip')

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
