import numpy as np

from algorithms.TD.TD0 import TD0


class DoubleQLearning(TD0):
    def __init__(self, env, gamma=1, epsilon=0.1, alpha=0.5):
        super().__init__(env, gamma, epsilon, alpha)
        self.Q2 = np.zeros(env.state_space + (env.action_space,))

    def estimate(self, n_episodes=1000):
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            state = self.env.reset()
            while True:
                action = self.policy(state)
                s_p, reward, done = self.env.step(action)
                rewards[i] += reward
                if np.random.rand() < 0.5:
                    a_p = np.argmax(self.Q[s_p])
                    self.Q[state+(action,)] += \
                        self.alpha * (reward + self.gamma * self.Q2[s_p + (a_p,)] - self.Q[state+(action,)])
                else:
                    a_p = np.argmax(self.Q2[s_p])
                    self.Q2[state + (action,)] += \
                        self.alpha * (reward + self.gamma * self.Q[s_p + (a_p,)] - self.Q2[state + (action,)])
                if done:
                    break
                state = s_p
        return rewards

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space)
        S = self.Q[state] + self.Q2[state]
        return np.random.choice(np.flatnonzero(S == S.max()))  # breaking ties randomly

    def reset(self):
        self.Q = np.zeros(self.env.state_space + (self.env.action_space,))
        self.Q2 = np.zeros(self.env.state_space + (self.env.action_space,))
