import numpy as np

class OUNoise():
    def __init__(self, n_actions, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
        self.theta = theta
        self.n_actions = n_actions
        self.mu = np.zeros(n_actions)
        self.sigma = sigma
        self.sqrt = np.sqrt(dt)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.n_actions)
        self.x_prev = x
        return x

    def reset(self):
        self.mu = np.zeros_like(self.mu)
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
