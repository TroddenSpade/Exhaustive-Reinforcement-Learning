from collections import defaultdict
import numpy as np
from tqdm import tqdm


class MCPrediction:
    def __init__(self, env, policy, gamma=1, episode_limit=10000, EVERY_VISIT=False):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.V = np.zeros((22, 11, 2))
        self.returns = np.zeros((22, 11, 2))
        self.N = np.zeros((22, 11, 2))
        self.EVERY_VISIT = EVERY_VISIT
        self.episode_limit = episode_limit

    def iterate(self, n_episodes=1000):
        for _ in tqdm(range(n_episodes)):
            states, _, rewards, length = self.generate_episode()
            G = 0
            for t in reversed(range(length)):
                G = self.gamma * G + rewards[t]
                if self.EVERY_VISIT or (states[t] not in states[:t]):
                    state = (states[t][0], states[t][1], int(states[t][2]))
                    self.N[state] += 1
                    self.V[state] += (G - self.V[state]) / self.N[state]    # Incremental Implementation

    def generate_episode(self):
        actions, states, rewards = [], [], []

        state = self.env.reset()
        i = 0
        for i in range(self.episode_limit):
            states.append(state)
            action = self.policy(state)
            actions.append(action)
            next_state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
            state = next_state
        return states, actions, rewards, i + 1

    def get_value(self):
        return self.V
