class WindyGridWorld:
    def __init__(self):
        self.state_space = (7, 10)
        self.action_space = 4
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.goal = (3, 7)
        self.player = (3, 0)

    def reset(self):
        self.player = (3, 0)
        return self.player

    def step(self, action):
        # (x, y)
        if action == 0:  # up
            x = max(self.player[0] - self.wind[self.player[1]] - 1, 0)
            self.player = (x, self.player[1])
        elif action == 1:  # right
            x = max(self.player[0] - self.wind[self.player[1]], 0)
            y = min(self.player[1] + 1, 9)
            self.player = (x, y)
        elif action == 2:  # down
            x = min(max(self.player[0] - self.wind[self.player[1]] + 1, 0), 6)
            self.player = (x, self.player[1])
        elif action == 3:  # left
            x = max(self.player[0] - self.wind[self.player[1]], 0)
            y = max(self.player[1] - 1, 0)
            self.player = (x, y)

        return self.player, -1, True if self.player == self.goal else False
