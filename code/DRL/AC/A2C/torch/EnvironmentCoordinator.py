import numpy as np
import torch.multiprocessing as mp

from .Env import Env

class EnvironmentCoordinator:
    def __init__(self, create_env, n_workers) -> None:
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for _ in range(n_workers)]

        self.workers = [
            Env(i, create_env, self.pipes[i][1]) 
            for i in range(self.n_workers)]
        [w.start() for w in self.workers]

    def reset(self):
        for coordinator_end, _ in self.pipes:
            coordinator_end.send(("reset", {}))
        return np.array([coordinator_end.recv() for coordinator_end, _ in self.pipes])

    def step(self, actions):
        for i, (coordinator_end, _) in enumerate(self.pipes):
            coordinator_end.send(('step', {'action':actions[i]}))

        states_p = []
        rewards = []
        dones = []
        is_terminals = []
        for i in range(self.n_workers):
            coordinator_end, _ = self.pipes[i]
            state_p, reward, done, info = coordinator_end.recv()
            states_p.append(state_p)
            rewards.append(reward)
            dones.append(done)
            is_terminals.append(done and not (
                'TimeLimit.truncated' in info and info['TimeLimit.truncated']))
        return states_p, rewards, dones, is_terminals 

    def close(self):
        for coordinator_end, _ in self.pipes:
            coordinator_end.send(("close", {}))
        [w.join() for w in self.workers]