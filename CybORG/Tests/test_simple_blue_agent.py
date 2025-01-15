import unittest
from agents.simple_blue_agent import SimpleBlueAgent

class TestSimpleBlueAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleBlueAgent()

    def test_isolate_host(self):
        observation = {'suspicious_activity': True}
        action = self.agent.take_action(observation)
        self.assertEqual(action, 'isolate_host')

    def test_terminate_process(self):
        observation = {'high_cpu_usage': True}
        action = self.agent.take_action(observation)
        self.assertEqual(action, 'terminate_process')

    def test_monitor(self):
        observation = {}
        action = self.agent.take_action(observation)
        self.assertEqual(action, 'monitor')

if __name__ == "__main__":
    unittest.main()
