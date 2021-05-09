import numpy as np


class SmallMDP:
    def __init__(self):
        self.state = 2
        self.B_states = 100
        self.state_space = (4,)
        self.action_space = self.B_states

    def reset(self):
        self.state = (2,)
        return self.state

    def step(self, action):
        reward = 0
        done = False
        if self.state == (2,):
            if action == 1:
                self.state = (3,)
                done = True
            else:
                self.state = (1,)
        elif self.state == (1,):
            reward = np.random.normal(-0.1, 1)
            done = True
            self.state = (0,)

        return self.state, reward, done
