import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

class DroneEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(DroneEnv, self).__init__()

        # Path to your MuJoCo XML model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'scene.xml')

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Renderer attributes
        self.render_mode = render_mode
        self.renderer = None

        # Define action space: thrust inputs for the four motors
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.11772, 0.11772, 0.11772, 0.11772], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space
        obs_low = np.full(13, -np.inf, dtype=np.float32)
        obs_high = np.full(13, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Get the ID of the goal site
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'goal_site')

        # Simulation parameters
        self.simulation_steps = 10

        # Seed the environment
        self.np_random = None
        self.seed()

        self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker')

    def _get_obs(self):
        # Get observations
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:7].copy()
        linear_velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()

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
        self.data.ctrl[:] = action

        # Step simulation
        for _ in range(self.simulation_steps):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        position = self.data.qpos[:3]
        distance = np.linalg.norm(position - self.target_position)
        reward = -distance

        # Distance to flat rotation
        orientation = self.data.qpos[3:7]
        angle = 2.0 * np.arccos(np.abs(orientation[0]))
        reward -= angle

        # Check if terminated or truncated
        terminated = False
        truncated = False
        if position[2] < 0.0 or position[2] > 2.0:
            terminated = True
            reward -= 100
        
        #check if out of bounds
        if np.linalg.norm(position) > 2.0:
            terminated = True
            reward -= 100


        # Additional info
        info = {
            'position': position.copy(),
            'distance_to_target': distance
        }

        # Render if necessary
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Handle seed
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset to hover keyframe if available
        keyframe_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(self.model.nkey)
        ]
        if 'hover' in keyframe_names:
            hover_key = keyframe_names.index('hover')
            mujoco.mj_resetDataKeyframe(self.model, self.data, hover_key)

        # Update the goal position if necessary
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.model.geom_pos[self.goal_geom_id] = self.target_position

        # Update the site position
        self.model.site_pos[self.goal_site_id] = self.target_position

        # Return initial observation and info
        obs = self._get_obs()
        info = {}

        # Render if necessary
        if self.render_mode == 'human':
            self.render()

        return obs, info

    def render(self):
        if self.render_mode == 'human':
            if self.renderer is None:
                # Initialize viewer
                self.renderer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                # Check if the viewer is still running
                if self.renderer.is_running():
                    # Update the renderer
                    with self.renderer.lock():
                        self.renderer.sync()
                else:
                    # Viewer has been closed by the user
                    self.renderer = None  # Reset the renderer
        elif self.render_mode == 'rgb_array':
            if self.renderer is None:
                # Initialize offscreen renderer
                self.renderer = mujoco.Renderer(self.model)
            # Render the scene
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
            return img
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
        if self.renderer is not None:
            # Close the renderer
            
            self.renderer.close()
            self.renderer = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
