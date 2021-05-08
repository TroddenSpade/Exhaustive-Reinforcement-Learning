import numpy as np

from algorithms.TD.TD0 import TD0


class QLearning(TD0):
    def estimate(self, n_episodes=1000):
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            state = self.env.reset()
            while True:
                action = self.policy(state)
                s_p, reward, done = self.env.step(action)
                rewards[i] += reward
                self.Q[state+(action,)] += \
                    self.alpha * (reward + self.gamma * np.max(self.Q[s_p]) - self.Q[state+(action,)])
                if done:
                    break
                state = s_p
        return rewards

