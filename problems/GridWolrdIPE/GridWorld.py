import numpy as np


class GridWorld:
    def __init__(self):
        self.n_states = 16
        self.n_actions = 4
        self.actions = range(self.n_actions)
        self.states = range(1, self.n_states-1)

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def pi(self, a, s):
        if a in range(4) and s in range(16):
            return 0.25
        return 0

    def P(self, s_p, r, s, a):
        if r != -1:
            return 0
        _, expected_s_p, _ = self.step(s, a)
        if expected_s_p != s_p:
            return 0
        return 1

    def step(self, state, action):
        if action == 0:  # Up
            s_p = state if state < 4 else state - 4
        elif action == 1:  # Right
            s_p = state if state % 4 == 3 else state + 1
        elif action == 2:  # Down
            s_p = state if state > 11 else state + 4
        elif action == 3:
            s_p = state if state % 4 == 0 else state - 1
        else:
            s_p = state

        return -1, s_p, True if s_p == 0 or s_p == 15 else False
