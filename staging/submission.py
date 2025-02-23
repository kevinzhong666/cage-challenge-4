import os
from ray.rllib.algorithms.ppo import PPO
from CybORG import CybORG
from CybORG.Agents.Wrappers import EnterpriseMAE

from blue_agent import env_creator_CC4, POLICY_MAP

class Submission:
    """Defines the trained agents for CAGE Challenge 4."""

    NAME = "MyAgent"
    TEAM = "MyTeam"
    TECHNIQUE = "PPO Multi-Agent"

    # Load the correct checkpoint directory
    checkpoint_path = os.path.join(os.path.dirname(__file__), "results")

    algo = PPO.from_checkpoint(checkpoint_path)

    # Load policies from trained model
    AGENTS = {
        "blue_agent_0": algo.get_policy("Agent0"),
        "blue_agent_1": algo.get_policy("Agent1"),
        "blue_agent_2": algo.get_policy("Agent2"),
        "blue_agent_3": algo.get_policy("Agent3"),
        "blue_agent_4": algo.get_policy("Agent4"),
    }

    @staticmethod
    def wrap(env: CybORG) -> EnterpriseMAE:
        """Wraps the environment with EnterpriseMAE for evaluation."""
        return EnterpriseMAE(env)
