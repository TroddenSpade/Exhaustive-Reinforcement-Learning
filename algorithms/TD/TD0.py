import numpy as np
from tqdm import tqdm

class TD0:
    def __init__(self, env, gamma=1, epsilon=0.1, alpha=0.5):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(env.state_space + (env.action_space,))

    def estimate(self, n_episodes=1000):
        pass

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space)
        return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))  # breaking ties randomly

    def stats(self, n_runs, n_episodes):
        rewards = np.zeros(n_episodes)
        for _ in tqdm(range(n_runs)):
            self.reset()
            rewards += self.estimate(n_episodes)
        return rewards / n_runs

    def reset(self):
        self.Q = np.zeros(self.env.state_space + (self.env.action_space,))

    def get_policy(self):
        return self.Q.argmax(axis=2)

    def get_value(self):
        return self.Q.max(axis=2)
