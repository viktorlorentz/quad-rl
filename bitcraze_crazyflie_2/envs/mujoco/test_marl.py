"""
mappo_multiquad.py

This file defines:
  - A dummy multi-quad environment (MultiquadEnv)
  - A multi-agent wrapper (MultiQuadMARLWrapper) that splits the observation and
    later combines the agents’ actions.
  - Actor and Critic network definitions using Flax:
      • Actor network: hidden layers [128, 64, 64]
      • Critic network: hidden layers [128, 128, 128, 128]
  - A main() function that runs MAPPO training via a hypothetical train_mappo API from JaxMARL.
  
Replace the dummy environment code with your actual environment implementation.
"""

import time
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
# The following imports assume that your JaxMARL installation provides these modules.
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.algorithms.mappo import train_mappo  # Hypothetical MAPPO training function


# ==========================================================
# 1. Define the base multi-quad environment.
#    (Replace this with your actual environment implementation.)
# ==========================================================

class MultiquadEnv:
    def __init__(self):
        # Example dimensions (modify as needed)
        self.action_dim = 4      # each quad's action dimension
        self.obs_dim = 8         # each quad's observation dimension
        # Total observation for both quads is 2 * obs_dim
       
    def reset(self, key):
        # For illustration, we return a flat observation vector of length 16 (8 per quad)
        obs = jax.random.normal(key, (self.obs_dim * 2,))
        state = {"dummy_state": 0}  # replace with your environment state
        return obs, state

    def step(self, state, action):
        # 'action' is expected to be a flat vector (length = 8)
        # Here we simulate a next observation and a reward
        next_obs = action * 0.1 + jnp.array([0.5] * (self.obs_dim * 2))
        reward = jnp.sum(action)  # dummy reward: sum of actions
        done = jnp.array(0.0)     # not terminal (for illustration)
        info = {}
        return next_obs, state, reward, done, info


# ==========================================================
# 2. Define the multi-agent wrapper.
#    This converts the flat observations/actions into dictionaries keyed by agent.
# ==========================================================

class MultiQuadMARLWrapper(JaxMARLWrapper):
    """
    Wraps a single-instance multi-quad environment into a multi-agent environment.
    Assumes the base environment returns a flat observation of length (2 * obs_dim)
    and expects a flat action of length (2 * action_dim).
    """
    def __init__(self, env):
        self._env = env
        self._num_agents = 2
        self._action_dim = env.action_dim
        self._obs_dim = env.obs_dim

    def reset(self, key):
        obs, state = self._env.reset(key)
        # Split the observation evenly between the two agents.
        obs_agent1 = obs[:self._obs_dim]
        obs_agent2 = obs[self._obs_dim:]
        obs_dict = {"quad1": obs_agent1, "quad2": obs_agent2}
        return obs_dict, state

    def step(self, state, action_dict):
        # Combine the actions from both agents.
        action = jnp.concatenate([action_dict["quad1"], action_dict["quad2"]])
        obs, state, reward, done, info = self._env.step(state, action)
        # Split the observation back for each agent.
        obs_agent1 = obs[:self._obs_dim]
        obs_agent2 = obs[self._obs_dim:]
        obs_dict = {"quad1": obs_agent1, "quad2": obs_agent2}
        return obs_dict, state, reward, done, info

    @property
    def num_agents(self):
        return self._num_agents

    def observation_space(self, agent):
        # Return the observation dimension for the given agent.
        return self._obs_dim

    def action_space(self, agent):
        # Return the action dimension for the given agent.
        return self._action_dim


# ==========================================================
# 3. Define the network architectures.
# ==========================================================

# Actor network with hidden layers: [128, 64, 64]
class ActorNetwork(nn.Module):
    action_dim: int  # dimensionality of the agent's action space

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        # Output the mean of a Gaussian policy.
        mean = nn.Dense(self.action_dim)(x)
        # Log standard deviation is a learned parameter.
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        return mean, std

# Critic network with hidden layers: [128, 128, 128, 128]
class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        # Output a single scalar value estimate.
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


# ==========================================================
# 4. Set up the training configuration and main training loop.
# ==========================================================

# Example configuration dictionary for MAPPO training.
config = {
    "ENV_NAME": "multiquad",              # name used for registration
    "NUM_AGENTS": 2,                      # two agents (one per quad)
    "NUM_ENVS": 1024,                     # number of parallel environment instances
    "TOTAL_TIMESTEPS": 250_000_000,         # total number of timesteps to train
    "EPISODE_LENGTH": 2000,               # episode length
    "ACTOR_HIDDEN_SIZES": [128, 64, 64],    # policy network sizes
    "CRITIC_HIDDEN_SIZES": [128, 128, 128, 128],  # critic network sizes
    "LEARNING_RATE": 3e-4,                # learning rate for both actor and critic
    "DISCOUNT": 0.99,                     # discount factor
    # ... include additional hyperparameters as needed
}

def main():
    # Create a PRNG key.
    key = jax.random.PRNGKey(0)

    # Instantiate the raw environment and wrap it.
    raw_env = MultiquadEnv()
    env = MultiQuadMARLWrapper(raw_env)

    # (Optional) Print environment info.
    print(f"Environment has {env.num_agents} agents, each with observation dim {env.observation_space('quad1')} and action dim {env.action_space('quad1')}.")

    # Start training using the MAPPO algorithm.
    # The train_mappo function is assumed to be provided by JaxMARL.
    start_time = time.time()
    trained_params, logs = train_mappo(
        env=env,
        actor_fn=ActorNetwork,
        critic_fn=CriticNetwork,
        config=config,
        rng=key,
    )
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save the trained parameters to a checkpoint.
    checkpoints.save_checkpoint(ckpt_dir="./checkpoints", target=trained_params, step=0, overwrite=True)
    print("Trained parameters saved in ./checkpoints/")

if __name__ == "__main__":
    main()