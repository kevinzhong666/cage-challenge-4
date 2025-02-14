import os
import time
import ray
import GPUtil

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from ray.tune import register_env
from ray.rllib.algorithms.mappo import MAPPOConfig
from ray.rllib.policy.policy import PolicySpec

# Ensure Ray starts locally (Remove this if using a cluster)
ray.init()

# Environment Creator for Challenge 4
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

# Register the environment with Ray
register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
env = env_creator_CC4({})

# Multi-agent setup
NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]

# RLlib Configuration
algo_config = (
    PPOConfig()
    .environment(env="CC4")
    .debugging(logger_config={"logdir": "logs/PPO_Example", "type": "ray.tune.logger.TBXLogger"})
    .framework("torch")  # Ensure PyTorch is used
    .resources(num_gpus=1)  # Allocate GPU for training
    .multi_agent(
        policies={
            ray_agent: PolicySpec(
                policy_class=None,  # RLlib default policy
                observation_space=env.observation_space(cyborg_agent),
                action_space=env.action_space(cyborg_agent),
                config={"gamma": 0.99},  # Higher discount factor for MARL
            ) for cyborg_agent, ray_agent in POLICY_MAP.items()
        },
        policy_mapping_fn=policy_mapper
    )
    .env_runners(
        num_env_runners=2,  # Increase number of parallel rollout workers
        num_envs_per_env_runner=4,  # Run more environments per worker
        rollout_fragment_length=100  # Adjust rollout length
    )
    .training()
    .update_from_dict({
        "train_batch_size": 24_000,  # Increase batch size for stability
        "sgd_minibatch_size": 6_144,  # Larger minibatches
        "num_sgd_iter": 20,  # More training iterations per batch
        "_use_fp16": True,  # Enable mixed precision
        "reuse_actors": True,  # Reduce overhead
        "vf_loss_coeff": 1.0,  # Value function loss coefficient
        "entropy_coeff": 0.02,  # Encourage exploration
        "clip_param": 0.2,  # Adjust PPO clipping for MAPPO
        "use_critic": True,  # Use centralized critic
        "use_gae": True,  # Enable Generalized Advantage Estimation (GAE)
        "lambda": 0.95,  # GAE discount factor
        "lr": 5e-4,  # Learning rate adjustment
    })
    .debugging(log_level="WARN")  # Reduce logging overhead for performance
)


# Function to log GPU usage
def log_gpu_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name}, Load: {gpu.load*100:.2f}%, Free Mem: {gpu.memoryFree}MB, Used Mem: {gpu.memoryUsed}MB")

# Train the RL algorithm
algo = algo_config.build()

def log_training_info(iteration, train_info):
    print("\n" + "="*50)
    print(f" Training Iteration: {iteration}")
    print("="*50)
    
    # Environment metrics
    env_info = train_info.get("env_runners", {})
    print(f" Environment Performance")
    print(f"  - Episode Reward (Mean): {env_info.get('episode_reward_mean', 'N/A')}")
    print(f"  - Episode Reward (Max/Min): {env_info.get('episode_reward_max', 'N/A')} / {env_info.get('episode_reward_min', 'N/A')}")
    print(f"  - Average Episode Length: {env_info.get('episode_len_mean', 'N/A')} timesteps")

    # Training stats per agent
    learner_info = train_info.get("info", {}).get("learner", {})
    print("\n Agent Performance")
    print(f"{'Agent':<10} {'Policy Loss':<15} {'Value Loss':<15} {'Entropy':<15} {'KL Divergence'}")
    print("-" * 65)

    for agent, stats in learner_info.items():
        learner_stats = stats.get("learner_stats", {})
        print(f"{agent:<10} {learner_stats.get('policy_loss', 'N/A'):<15.5f} "
              f"{learner_stats.get('vf_loss', 'N/A'):<15.5f} "
              f"{learner_stats.get('entropy', 'N/A'):<15.5f} "
              f"{learner_stats.get('kl', 'N/A')}")

    # Timing and performance
    timing_info = train_info.get("timers", {})
    print("\n Performance Stats")
    print(f"  - Training Step Time: {timing_info.get('training_step_time_ms', 'N/A')} ms")
    print(f"  - Inference Time per Step: {env_info.get('sampler_perf', {}).get('mean_inference_ms', 'N/A')} ms")
    print(f"  - Environment Wait Time: {env_info.get('sampler_perf', {}).get('mean_env_wait_ms', 'N/A')} ms")
    print(f"  - Steps per Second: {env_info.get('num_env_steps_sampled_throughput_per_sec', 'N/A')}")

    print("="*50 + "\n")

# Run training and log results
for i in range(50):
    train_info = algo.train()
    log_training_info(i, train_info)
    log_gpu_usage()

    # Save every 10 iterations
    if i % 10 == 0:
        algo.save(f"results/MAPPO_checkpoint_{i}")

# Save final trained model
algo.save("results/MAPPO_final")

print("MAPPO training complete. Check 'results' directory for checkpoints.")