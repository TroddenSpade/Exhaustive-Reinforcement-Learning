import numpy as np

from algorithms.TD.TD0 import TD0


class ExpectedSARSA(TD0):
    def estimate(self, n_episodes=1000):
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            state = self.env.reset()
            while True:
                action = self.policy(state)
                s_p, reward, done = self.env.step(action)
                rewards[i] += reward

                a_p = np.argmax(self.Q[state])
                pi = np.ones(self.env.action_space) * (self.epsilon / self.env.action_space)
                pi[a_p] += 1 - self.epsilon

                self.Q[state+(action,)] += \
                    self.alpha * (reward + self.gamma * np.sum(pi * self.Q[s_p]) - self.Q[state+(action,)])
                if done:
                    break
                state = s_p
        return rewards
