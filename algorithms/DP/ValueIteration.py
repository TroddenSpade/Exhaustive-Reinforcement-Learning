import numpy as np
from tqdm import tqdm


class ValueIteration:
    def __init__(self, Env, theta=0.1, gamma=1):
        self.env = Env()
        self.V = np.zeros(self.env.state_space)
        self.gamma = gamma
        self.theta = theta
        self.pi = np.zeros(self.env.state_space)

    def iterate(self):
        done = False
        pbar = tqdm()
        while not done:
            pbar.update(1)
            new_V = np.zeros(self.env.state_space)
            for s in self.env.get_states():
                actions = self.env.get_actions(s)
                q = np.zeros(len(actions))
                for a in actions:
                    list = self.env.pp(s, a)
                    for (P, reward, s_p) in list:
                        q[a] += P * (reward + self.gamma * self.V[s_p])
                new_V[s] = np.max(q)
                self.pi[s] = np.argmax(q)
            delta = np.max(np.abs(new_V - self.V))
            self.V = new_V
            if delta < self.theta:
                done = True
        pbar.close()

    def value(self):
        return self.V[0:self.env.state_space-1]

    def policy(self):
        for s in self.env.get_states():
            actions = self.env.get_actions(s)
            q = np.zeros(len(actions))
            for a in actions:
                list = self.env.pp(s, a)
                for (P, reward, s_p) in list:
                    q[a] += P * (reward + self.gamma * self.V[s_p])
            self.pi[s] = np.argmax(q)
        return self.pi
