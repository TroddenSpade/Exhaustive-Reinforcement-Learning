import gym

class DefineEnv:
    def __init__(self, name):
        self.name = name
    
    def __call__(self):
        return gym.make(self.name)