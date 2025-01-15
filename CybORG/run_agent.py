import sys
import os

# Add the base directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.simple_blue_agent import SimpleBlueAgent

def main():
    agent = SimpleBlueAgent()
    example_observation = {'suspicious_activity': True}  # Example input
    action = agent.take_action(example_observation)
    print(f"Agent action: {action}")

if __name__ == "__main__":
    main()
