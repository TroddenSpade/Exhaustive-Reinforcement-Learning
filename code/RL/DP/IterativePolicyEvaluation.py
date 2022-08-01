import numpy as np
from tqdm import tqdm

class IterativePolicyEvaluation:
    def __init__(self, _ENV_, theta, gamma):
        self.env = _ENV_()
        self.V = np.zeros(self.env.n_states)
        self.states = self.env.get_states()
        self.actions = self.env.get_actions()
        self.theta = theta
        self.gamma = gamma
        self.delta = 0

    def evaluate(self):
        done = False
        pbar = tqdm()
        while not done:
            pbar.update(1)
            new_V = np.zeros(self.env.n_states)
            for s in self.states:
                for a in self.actions:
                    reward, s_p, _ = self.env.step(s, a)
                    new_V[s] += self.env.pi(a, s) * self.env.P(s_p, reward, s, a) * (reward + self.gamma * self.V[s_p])
            delta = np.max(np.abs(self.V - new_V))
            self.V = new_V
            if delta < self.theta:
                done = True
        pbar.close()

    def get_V(self):
        return self.V

