import numpy as np


class GamblersProblem:
    def __init__(self):
        self.action_space = 101
        self.state_space = 101
        self.p_h = 0.4

    def get_states(self):
        return range(1, self.state_space-1)

    def get_actions(self, s):
        return range(0, min(s, self.state_space -1 - s) + 1)

    def step(self, state, action):
        pass

    def pp(self, state, action):
        list = []
        if state + action == self.state_space -1:
            list.append((self.p_h, 1, state + action))
        else:
            list.append((self.p_h, 0, state + action))

        list.append((1-self.p_h, 0, state - action))

        return list

