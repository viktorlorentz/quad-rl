import os
import gymnasium as gym
from stable_baselines3 import PPO
import bitcraze_crazyflie_2  # Ensure your environment is registered
import torch as th
import torch.nn as nn
import onnxruntime as ort
import numpy as np


class OnnxablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        # Extract features using the policy's features extractor
        features = self.policy.extract_features(observation)
        # Pass through the policy network (actor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)
        # Return the mean actions (for deterministic policy)
        return mean_actions


def main():
    # Create the environment with rendering enabled
    env = gym.make("DroneEnv-v0", render_mode="human")

    # Path to the saved model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    model_path = os.path.join(models_dir, "model.zip")

    # Load the trained model
    model = PPO.load(model_path, device="cpu")

    # Export only the policy (actor network) to ONNX
    onnx_policy = OnnxablePolicy(model.policy)

    observation_shape = model.observation_space.shape
    dummy_input = th.randn(1, *observation_shape)

    onnx_model_path = "my_ppo_actor.onnx"
    th.onnx.export(
        onnx_policy,
        dummy_input,
        onnx_model_path,
        opset_version=17,
        input_names=["input"],
        output_names=["action"],
        dynamic_axes={"input": {0: "batch_size"}, "action": {0: "batch_size"}},
    )

    # Load the ONNX model
    ort_sess = ort.InferenceSession(onnx_model_path)

    # Evaluate the ONNX model
    obs, info = env.reset()
    terminated = False
    truncated = False

    last_reward = 0.0

    while not (terminated or truncated):
        # Preprocess observation
        if isinstance(obs, tuple):
            obs = obs[0]  # Unpack if observation is a tuple
        observation = np.array(obs, dtype=np.float32)
        observation = observation.reshape(1, *observation.shape)  # Add batch dimension

        # Get action from ONNX model
        mean_actions = ort_sess.run(None, {"input": observation})[0]
        scaled_action = mean_actions[0]

        # Post-process action
        # Rescale the action from [-1, 1] to [low, high]
        low, high = env.action_space.low, env.action_space.high
        action = low + (0.5 * (scaled_action + 1.0) * (high - low))
        action = np.clip(action, low, high)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = reward

    print(f"Final reward: {last_reward}")

    env.close()


if __name__ == "__main__":
    main()
