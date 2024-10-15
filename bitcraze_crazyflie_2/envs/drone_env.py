import gym
from gym import spaces
import numpy as np
import mujoco
import os

class DroneEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(DroneEnv, self).__init__()

        # Path to your MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'cf2.xml')

        # Load the MuJoCo model
        self.model = mujoco.load_model_from_path(model_path)
        self.sim = mujoco.MjSim(self.model)

        # Viewer is optional; initialize it only if you plan to render
        self.viewer = None

        # Define action space: thrust inputs for the four motors
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.11772, 0.11772, 0.11772, 0.11772]),
            dtype=np.float32
        )

        # Define observation space
        obs_low = np.array([-np.inf] * 13, dtype=np.float32)
        obs_high = np.array([np.inf] * 13, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.0])

        # Simulation parameters
        self.simulation_steps = 10

    def _get_obs(self):
        # Get observations
        position = self.sim.data.qpos[:3]
        orientation = self.sim.data.qpos[3:7]
        linear_velocity = self.sim.data.qvel[:3]
        angular_velocity = self.sim.data.qvel[3:6]

        # Combine all observations
        obs = np.concatenate([
            position,
            orientation,
            linear_velocity,
            angular_velocity
        ])

        return obs.astype(np.float32)

    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action
        self.sim.data.ctrl[:] = action

        # Step simulation
        for _ in range(self.simulation_steps):
            self.sim.step()

        # Get observation
        obs = self._get_obs()

        # Compute reward
        position = self.sim.data.qpos[:3]
        distance = np.linalg.norm(position - self.target_position)
        reward = -distance

        # Check if done
        done = False
        if position[2] < 0.0 or position[2] > 2.0:
            done = True
            reward -= 100

        # Additional info
        info = {
            'position': position,
            'distance_to_target': distance
        }

        return obs, reward, done, info

    def reset(self):
        # Reset simulation
        self.sim.reset()

        # Reset to hover keyframe if available
        keyframe_names = [self.model.key_names[i].decode('utf-8') for i in range(self.model.nkey)]
        if 'hover' in keyframe_names:
            hover_key = keyframe_names.index('hover')
            self.sim.data.qpos[:] = self.model.key_qpos[hover_key]
            self.sim.data.qvel[:] = self.model.key_qvel[hover_key]
            self.sim.forward()

        # Return initial observation
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                # Initialize viewer
                self.viewer = mujoco.MjViewer(self.sim)
            self.viewer.render()
        elif mode == 'rgb_array':
            # Return RGB array
            width, height = 640, 480
            if self.viewer is None:
                self.viewer = mujoco.MjRenderContextOffscreen(self.sim, -1)
            self.viewer.render(width, height)
            data = self.viewer.read_pixels(width, height, depth=False)
            return data
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def close(self):
        if self.viewer is not None:
            # Clean up viewer
            self.viewer = None
