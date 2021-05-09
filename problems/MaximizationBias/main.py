from algorithms.TD.DoubleQLearning import DoubleQLearning
from algorithms.TD.QLearning import QLearning
from problems.MaximizationBias.SmallMDP import SmallMDP

import numpy as np
import matplotlib.pyplot as plt


class QL(QLearning):
    def policy(self, state):
        if state == (2,):
            if np.random.rand() < self.epsilon:
                return np.random.randint(2)
            return np.random.choice(np.flatnonzero(self.Q[state, :2] == self.Q[state, :2].max()))
        elif state == (1,):
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.env.action_space)
            return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))  # breaking ties randomly

class DQL(DoubleQLearning):
    def policy(self, state):
        if state == (2,):
            if np.random.rand() < self.epsilon:
                return np.random.randint(2)
            S = self.Q[state, :2] + self.Q2[state, :2]
            return np.random.choice(np.flatnonzero(S == S.max()))  # breaking ties randomly
        elif state == (1,):
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.env.action_space)
            S = self.Q[state] + self.Q2[state]
            return np.random.choice(np.flatnonzero(S == S.max()))  # breaking ties randomly


env = SmallMDP()

ql = QL(env)
dql = DQL(env)

rew_ql = ql.stats(10000, 300)
rew_dql = dql.stats(10000, 300)

plt.plot(rew_ql, 'deepskyblue', label="Q-Learning")
plt.plot(rew_dql, 'crimson', label="Double Q-Learning")
plt.legend()
plt.show()
