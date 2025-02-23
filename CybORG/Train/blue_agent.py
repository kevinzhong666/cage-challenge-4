import sys
sys.path.append("C:/Users/tarnv/Documents/Capstone/C4/cage-challenge-4")

import os
import time
import ray
import GPUtil
import torch
import torch.nn as nn
from datetime import datetime

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog

# if __name__ == "__main__":
#     if ray.is_initialized():
#         ray.shutdown()
#     ray.init(ignore_reinit_error=True)

def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    env = EnterpriseMAE(env=cyborg)
    return env

register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
env = env_creator_CC4({})

# if __name__ == "__main__":
#     print("Testing Environment...")
#     obs = env.reset()
    
#     for step in range(500):
#         if isinstance(obs, dict):
#             action = {agent: env.action_space(agent).sample() for agent in obs.keys()}
#         elif isinstance(obs, tuple):
#             obs_dict = obs[0] if isinstance(obs[0], dict) else obs
#             action = {agent: env.action_space(agent).sample() for agent in obs_dict.keys()}
#         else:
#             raise ValueError(f"Unexpected obs type: {type(obs)}")
        
#         step_output = env.step(action)

#         # Handle different output lengths
#         if len(step_output) == 4:
#             obs, rewards, dones, infos = step_output
#         elif len(step_output) == 5:
#             obs, rewards, dones, infos, _ = step_output  # Ignore the extra value
#         else:
#             raise ValueError(f"Unexpected step output format: {step_output}")

#     print("Environment Test Complete\n")

NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]

class CentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.fc = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )
        self.value_fn = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        self._last_obs = input_dict["obs"]
        x = self.fc(self._last_obs)
        return x, state

    def value_function(self):
        if self._last_obs is None:
            raise ValueError("No observation available to compute value function.")
        return torch.reshape(self.value_fn(self._last_obs), [-1])

ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

algo_config = (
    PPOConfig()
    .environment(env="CC4")
    .framework("torch")
    .resources(num_gpus=1)
    .env_runners(rollout_fragment_length=500)
    .reporting(keep_per_episode_custom_metrics=True)
    .training(lr=1e-4)
    .training(entropy_coeff=0.01)
    .multi_agent(
        policies={ray_agent: PolicySpec(
            policy_class=None,
            observation_space=env.observation_space(cyborg_agent),
            action_space=env.action_space(cyborg_agent),
            config={"gamma": 0.99, "model": {"custom_model": "centralized_critic"}},
        ) for cyborg_agent, ray_agent in POLICY_MAP.items()},
        policy_mapping_fn=policy_mapper
    )
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
)

def log_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            print(f"🔹 [GPU] {gpu.name} | Load: {gpu.load*100:.2f}% | Free Mem: {gpu.memoryFree}MB | Used Mem: {gpu.memoryUsed}MB")
    else:
        print("No GPUs detected.")

if __name__ == "__main__":
    algo = algo_config.build_algo()
    print("Starting Training...\n")

    for i in range(50):
        train_info = algo.train()
        print(f"\nFULL `train_info` OUTPUT (Training Iteration {i}):")
        

        #Extract rewards correctly
        avg_reward = train_info.get("env_runners", {}).get("episode_reward_mean", "N/A")
        max_reward = train_info.get("env_runners", {}).get("episode_reward_max", "N/A")
        min_reward = train_info.get("env_runners", {}).get("episode_reward_min", "N/A")

        print(f"\n🔹 Training Iteration {i} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Avg Reward: {avg_reward}")
        print(f"Max Reward: {max_reward}")
        print(f"Min Reward: {min_reward}")
        print(f"Steps Sampled: {train_info.get('num_env_steps_sampled', 0)} | Steps Trained: {train_info.get('num_env_steps_trained', 0)}")

        log_gpu_usage()

    algo.save("results")
    print("\nTraining complete. Results saved in 'results' directory.")