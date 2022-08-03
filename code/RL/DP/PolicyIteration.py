import numpy as np
from tqdm import tqdm


class PolicyIteration:
    def __init__(self, _ENV_, theta, gamma):
        self.env = _ENV_()
        self.V = np.zeros(self.env.n_states)
        self.pi = np.zeros(self.env.n_states)
        self.states = self.env.get_states()
        self.actions = self.env.get_actions()
        self.theta = theta
        self.gamma = gamma
        self.delta = 0

    def evaluation(self):
        done = False
        while not done:
            delta = 0
            new_V = np.zeros(self.env.n_states)
            for s in tqdm(self.states):
                a = self.pi[s]
                new_V[s] = self.value(s, a)

            self.V = new_V
            print(new_V)
            delta = max(delta, np.max(np.abs(self.V - new_V)))
            if delta < self.theta:
                done = True


    def improvement(self):
        new_pi = np.zeros(self.env.n_states)

        for s in self.states:
            q = np.zeros(self.env.n_actions)
            for a in self.actions:
                q[a] = self.value(s, a)
            new_pi[s] = np.argmax(q)

        # (self.pi - new_pi).all()
        if np.max(np.abs(self.pi - new_pi)) == 0:
            return True
        self.pi = new_pi
        return False

    def value(self, s, a):
        a = a - 5
        loc = [min(max(s // 21 + a, 0), 20), min(max(s % 21 - a, 0), 20)]
        new_V = 0
        for ret1 in range(11):
            for rent1 in range(11):
                for ret2 in range(11):
                    for rent2 in range(11):
                        req1 = min(loc[0], rent1)
                        req2 = min(loc[1], rent2)
                        reward = (req1 + req2) * 10 - abs(a) * 2
                        s_p = int(min(max(loc[0] - req1 + ret1, 0), 20) * 21 + min(max(loc[0] - req2 + ret2, 0), 20))
                        P = self.env.dist_l1[ret1, rent1] * self.env.dist_l2[ret2, rent2]
                        new_V += P * (reward + self.gamma * self.V[s_p])
        return new_V

    def iteration(self):
        done = False
        # pbar = tqdm()
        # while not done:
        # pbar.update(1)
        self.evaluation()
        done = self.improvement()
        # pbar.close()
        return self.pi

    def get_V(self):
        return self.V
