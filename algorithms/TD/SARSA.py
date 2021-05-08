import numpy as np

from algorithms.TD.TD0 import TD0


class SARSA(TD0):
    def estimate(self, n_episodes=1000):
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            state = self.env.reset()
            action = self.policy(state)
            while True:
                s_p, reward, done = self.env.step(action)
                rewards[i] += reward
                a_p = self.policy(s_p)
                self.Q[state+(action,)] += self.alpha * (reward + self.gamma * self.Q[s_p+(a_p,)] - self.Q[state+(action,)])
                if done:
                    break
                action, state = a_p, s_p
        return rewards