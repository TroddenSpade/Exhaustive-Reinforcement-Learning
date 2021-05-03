import numpy as np
from tqdm import tqdm


class GradientBandit:
    def __init__(self, n_actions, initial_q=0, policy='epsilon-greedy', alpha=None, epsilon=0.01, steps=2000):
        self.obtained_rewards = np.zeros(steps)
        self.H = np.zeros(n_actions)  # initialize preferences
        self.alpha = alpha
        self.n_actions = n_actions
        self.steps = steps
        self.policy_name = policy
        self.initial_q = initial_q
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.obtained_rewards = np.zeros(self.steps)
        self.H = np.zeros(self.n_actions)  # initialize preferences

    def train(self, _bandit_):
        b = _bandit_(self.n_actions)  # bandit class
        for s in range(self.steps):
            action = self.policy()
            reward = b.step(action)

            self.obtained_rewards[s] = reward

            r_t = np.mean(self.obtained_rewards[0:s])

            pi = self.softmax_pi()
            self.H[self.H != action] -= self.alpha * (reward - r_t) * pi[self.H != action]
            self.H[action] += self.alpha * (reward - r_t) * (1 - pi[action])

    def predict(self):
        return self.policy()

    def stats(self, iterations, _bandit_):
        rewards = np.zeros(self.steps)
        for _ in tqdm(range(iterations)):
            self.train(_bandit_)
            rewards += self.obtained_rewards
            self.reset()
        return rewards / iterations

    def softmax_pi(self):
        M = np.exp(self.H)
        return M / np.sum(M)

    def policy(self):
        M = np.exp(self.H)
        return np.random.choice(np.flatnonzero(M == M.max()))  # breaking ties randomly
