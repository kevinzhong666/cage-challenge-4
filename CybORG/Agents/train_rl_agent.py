from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from CybORG import CybORG
from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from gymnasium import Env, spaces
import numpy as np

# Define scenario parameters
EPISODE_LENGTH = 500
sg = EnterpriseScenarioGenerator(steps=EPISODE_LENGTH)

# Create the base CybORG environment
cyborg = CybORG(scenario_generator=sg, seed=1234)
base_env = cyborg  # Retain reference to the original CybORG environment

# Wrap the base environment with BlueFlatWrapper
blue_flat_env = BlueFlatWrapper(base_env)

# Create a Gym-compatible wrapper for BlueFlatWrapper
class GymCompatibleBlueFlatWrapper(Env):
    def __init__(self, cyborg_env):
        super(GymCompatibleBlueFlatWrapper, self).__init__()
        self.env = cyborg_env

        # Use action labels to determine the size of the action space
        self.action_space = spaces.Discrete(len(self.env.actions("blue_agent_0")))

        # Define observation space
        sample_obs = self.env.reset()[0]["blue_agent_0"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)  # Set the random seed if provided
        obs, info = self.env.reset()
        return np.array(obs["blue_agent_0"]), info

    def step(self, action):
        actions = {"blue_agent_0": action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return (
            np.array(obs["blue_agent_0"]),
            rewards["blue_agent_0"],
            terminated["blue_agent_0"],
            truncated["blue_agent_0"],
            info,
        )

# Wrap the BlueFlatWrapper with the Gym-compatible wrapper
gym_env = GymCompatibleBlueFlatWrapper(blue_flat_env)
vec_env = DummyVecEnv([lambda: gym_env])

# Initialize PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("blue_agent_rl")
print("Model training complete and saved as blue_agent_rl.zip")
