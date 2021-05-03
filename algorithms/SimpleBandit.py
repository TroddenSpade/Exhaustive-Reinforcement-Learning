import numpy as np
from tqdm import tqdm


class SimpleBandit:
    def __init__(self, n_actions, initial_q, policy, alpha=None, epsilon=0.01, steps=2000):
        self.obtained_rewards = np.zeros(steps)
        self.N = np.zeros(n_actions)  # initialize number of rewards given
        self.Q = np.ones(n_actions) * initial_q  # initial Q
        self.alpha = alpha
        self.n_actions = n_actions
        self.steps = steps
        self.policy_name = policy
        self.initial_q = initial_q
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        # self.selected_actions = np.zeros(steps)
        self.obtained_rewards = np.zeros(self.steps)
        self.N = np.zeros(self.n_actions)
        self.Q = np.ones(self.n_actions) * self.initial_q

    def train(self, _bandit_):
        b = _bandit_(self.n_actions)  # bandit class
        for s in range(self.steps):
            action = self.policy(self.epsilon, s + 1)
            reward = b.step(action)

            self.obtained_rewards[s] = reward

            self.N[action] += 1
            if self.alpha:
                self.Q[action] += self.alpha * (reward - self.Q[action])
            else:
                self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def predict(self):
        return np.argmax(self.Q)

    def stats(self, iterations, _bandit_):
        rewards = np.zeros(self.steps)
        for _ in tqdm(range(iterations)):
            self.train(_bandit_)
            rewards += self.obtained_rewards
            self.reset()
        return rewards / iterations

    def policy(self, epsilon, t):
        if self.policy_name == 'ucb':
            return self.__upper_confidence_bound(t)
        else:
            return self.__epsilon_greedy(epsilon)

    def __epsilon_greedy(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))  # breaking ties randomly

    def __upper_confidence_bound(self, t):
        c = 2
        if self.N.min() == 0:
            return np.random.choice(np.flatnonzero(self.N == self.N.min()))
        M = self.Q + c * np.sqrt(np.divide(np.log(t), self.N))
        return np.argmax(M)
