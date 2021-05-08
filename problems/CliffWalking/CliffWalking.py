class CliffWalking:
    def __init__(self):
        self.state_space = (4, 12)
        self.action_space = 4
        self.cliff = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
        self.goal = (3, 11)
        self.init = (3, 0)
        self.player = self.init

    def reset(self):
        self.player = self.init
        return self.player

    def step(self, action):
        # (x, y)
        reward = -1
        if action == 0:  # up
            x = max(self.player[0] - 1, 0)
            self.player = (x, self.player[1])
        elif action == 1:  # right
            y = min(self.player[1] + 1, 11)
            if (self.player[0], y) in self.cliff:
                reward = -100
                self.player = self.init
            else:
                self.player = (self.player[0], y)
        elif action == 2:  # down
            x = min(self.player[0] + 1, 3)
            if (x, self.player[1]) in self.cliff:
                reward = -100
                self.player = self.init
            else:
                self.player = (x, self.player[1])
        elif action == 3:  # left
            y = max(self.player[1] - 1, 0)
            self.player = (self.player[0], y)

        return self.player, reward, True if self.player == self.goal else False
