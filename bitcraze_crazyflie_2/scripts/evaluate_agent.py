import gym
import os
from stable_baselines3 import PPO
import bitcraze_crazyflie_2  # Ensure your environment is registered

def main():
    # Create the environment
    env = gym.make('DroneEnv-v0')

    # Path to the saved model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    model_path = os.path.join(models_dir, 'ppo_drone.zip')

    # Load the trained model
    model = PPO.load(model_path)

    # Evaluate the model
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()

if __name__ == '__main__':
    main()
