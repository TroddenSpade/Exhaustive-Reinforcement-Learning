import numpy as np
from tqdm import tqdm


class MCEpsilonSoftControl:
    def __init__(self, env, epsilon=0.1, gamma=1, episode_limit=10000):
        self.env = env
        self.pi = np.zeros((22, 11, 2, 2))
        self.N = np.zeros((22, 11, 2, 2))
        self.Q = np.zeros((22, 11, 2, 2))
        self.action_space = 2
        self.epsilon = epsilon
        self.episode_limit = episode_limit
        self.gamma = gamma

    def iterate(self, n_episodes=1000):
        for _ in tqdm(range(n_episodes)):
            state_action, rewards, length = self.generate_episode()
            G = 0
            for t in reversed(range(length)):
                G = rewards[t] + self.gamma * G
                if state_action[t] not in state_action[:t]:
                    self.N[state_action[t]] += 1
                    self.Q[state_action[t]] += (G - self.Q[state_action[t]]) / self.N[state_action[t]]
                    a_star = np.argmax(self.Q[state_action[t][:3]])
                    self.pi[state_action[t][:3]][:a_star] = self.epsilon / self.action_space
                    self.pi[state_action[t][:3]][a_star+1:] = self.epsilon / self.action_space
                    self.pi[state_action[t][:3]][a_star] = 1 - self.epsilon + self.epsilon / self.action_space

    def generate_episode(self):
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

    def get_policy(self):
        return self.pi.argmax(axis=3)

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
