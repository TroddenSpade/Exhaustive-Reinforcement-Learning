import numpy as np

class Bandit:
    def __init__(self, n_actions):
        self.q_values = np.random.normal(0, 1, n_actions)

    def step(self, action):
        return np.random.normal(self.q_values[action], 1)
