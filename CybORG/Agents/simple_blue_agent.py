class SimpleBlueAgent:
    def __init__(self):
        print("SimpleBlueAgent initialized.")
    
    def take_action(self, observation):
        if 'suspicious_activity' in observation:
            return 'isolate_host'
        elif 'high_cpu_usage' in observation:
            return 'terminate_process'
        else:
            return 'monitor'
