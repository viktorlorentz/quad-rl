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
            low=np.zeros(4, dtype=np.float32),
            high=np.full(4, 0.11772, dtype=np.float32),
            dtype=np.float32
        )

        # Update observation space to include position error
        obs_low = np.full(16, -np.inf, dtype=np.float32)
        obs_high = np.full(16, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Set the target position for hovering
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Get the ID of the goal site
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'goal_site')

        # Get the drone's body ID
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cf2')

        # Simulation parameters
        self.simulation_steps = 1 # 250Hz

        self.workspace = {
            'low': np.array([-3.0, -3.0, 0.0]),
            'high': np.array([3.0, 3.0, 2.5])
        }

        # Seed the environment
        self.np_random = None
        self.seed()

        # Get the drone start position
        self.start_position = self.data.qpos[:3].copy()
        

        self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker')

    def _get_obs(self):
        # Get observations
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:7].copy()
        linear_velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()

        # Compute position error
        position_error = self.target_position - position

        # Combine all observations, including the position error
        obs = np.concatenate([
            position,
            orientation,
            linear_velocity,
            angular_velocity,
            position_error  # Include position error
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

        # Extract position and orientation
        position = self.data.qpos[:3]
        orientation = self.data.qpos[3:7]  # Quaternion [w, x, y, z]

        # Compute distance to target position
        distance = np.linalg.norm(position - self.target_position)

        
        # Compute rotation penalty
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Upright orientation

        # Compute dot product between current orientation and desired orientation
        dot_product = np.abs(np.dot(orientation, desired_orientation))
        # Ensure the dot product is within valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Compute the angle between orientations (in radians)
        angle = 2 * np.arccos(dot_product)
        rotation_penalty = angle  # Penalty proportional to the angle

        # Compute angular velocity penalty
        angular_velocity = self.data.qvel[3:6]
        angular_velocity_penalty = np.linalg.norm(angular_velocity)

        # Velocity penalty
        velocity_penalty = np.linalg.norm(self.data.qvel[:3])

        # Check if terminated or truncated
        terminated = False
        truncated = False

        reward = 4 # alive reward
        
        # Compute total reward
        reward -= distance # Reward proportional to the progress to the target

        # Gaussian reward bonus in goal region
        if distance < 0.1:
            reward += 4 * np.exp(-(distance)**2 / 0.01)


        # Subtract penalties
        reward -= 0.5 * rotation_penalty
        reward -= 0.05 * angular_velocity_penalty**2
        reward -= 0.01 * velocity_penalty**2

        
        

        
        # print( "rotation_penalty: ", rotation_penalty)
        # print( "angular_velocity_penalty: ", 0.1 * angular_velocity_penalty**2)
        # print( "distance: ", 2- distance)
        # print( "reward: ", reward)


        # Check for any contacts involving the drone's body
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get the body IDs involved in the contact
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            # Check if the drone's body is involved
            if body1 == self.drone_body_id or body2 == self.drone_body_id:
                collision = True
                break  # Exit the loop if a collision is detected
        
        if collision:
            terminated = True
            reward -= 1000



        # Normalizing actions to [0, 1]
        a = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)

        k=300

        dgb = (np.exp(-k * (a- 0)**2) + np.exp(-k * (a- 1)**2)) / (1 + np.exp(-k))

        # sum all 4 actions
        reward -= np.sum(dgb)/4

        
        




        #check if out of bounds
        if not np.all(self.workspace['low'] <= position) or not np.all(position <= self.workspace['high']):
            terminated = True
            reward -= 1000

        # Additional info
        info = {
            'position': position.copy(),
            'distance_to_target': distance,
            'rotation_penalty': rotation_penalty,
            'angular_velocity_penalty': angular_velocity_penalty
        }

        # Render if necessary
        if self.render_mode == 'human':
            self.render()

        # truncate episode if too long
        if self.data.time > 20:
            terminated = True
            truncated = True


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

        # add normal noise to the start position
        noise = self.np_random.normal(0, 0.1, size=3)
        self.data.qpos[:3] = self.data.qpos[:3].copy() + noise

        # set Start position
        self.start_position = self.data.qpos[:3].copy()


        # Update the target position if necessary
        self.target_position = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Update the goal marker position if you're displaying it
        self.model.site_pos[self.goal_site_id] = self.target_position
        mujoco.mj_forward(self.model, self.data)

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
