import numpy as np
from tqdm import tqdm


class OffPolicyMCControl:
    def __init__(self, env, epsilon=0.1, gamma=1, episode_limit=10000):
        self.env = env
        self.b = np.zeros((22, 11, 2, 2))
        self.Q = np.zeros((22, 11, 2, 2))
        self.pi = self.Q.max(axis=3)
        self.C = np.zeros((22, 11, 2, 2))
        self.action_space = 2
        self.epsilon = epsilon
        self.episode_limit = episode_limit
        self.gamma = gamma

    def iterate(self, n_episodes=1000):
        for _ in tqdm(range(n_episodes)):
            state_action, rewards, length = self.generate_episode_exploring_start()
            G = 0
            W = 1
            for t in reversed(range(length)):
                G = rewards[t] + self.gamma * G
                self.C[state_action] += W
                self.Q[state_action] += (G - self.Q[state_action]) * W / self.C[state_action]
                self.pi[state_action[:3]] = np.argmax(self.Q[state_action])
                if state_action[3] != self.pi[state_action[:3]]:
                    break
                W /= self.b[state_action]

    def generate_episode_exploring_start(self):
        state_action, rewards = [], []
        state = self.env.reset()

        i = 0
        for i in range(self.episode_limit):
            state = (state[0], state[1], int(state[2]))
            action = self.policy(state)
            state_action.append(state + (action,))
            next_state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
            state = next_state
        return state_action, rewards, i + 1

    def policy(self, state):
        if np.random.rand() < 1 - self.epsilon:
            return np.argmax(self.pi[state])
        return np.random.randint(0, self.action_space)

    def get_value(self):
        return self.Q.max(axis=3)

    @staticmethod
    def generate_state():
        player = np.random.randint(12, 22)
        dealer = np.random.randint(1, 11)
        ace = bool(np.random.randint(0, 2))
        return player, dealer, ace

    @staticmethod
    def generate_action():
        return np.random.randint(0, 2)
