import time
import pickle
from torch import functional
from tqdm import tqdm

import gym
import torch
import numpy as np
from torch import tensor
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


#fully-connected policy value network
class FCPV:
    def __init__(self, input_size, output_size, hidden_layers,
                 activation_fn=torch.nn.functional.relu,
                 optimizer=torch.optim.Adam, learning_rate=0.0005) -> None:
        self.activation_fn = activation_fn
        self.input_layer = torch.nn.Linear(input_size, hidden_layers[0])
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_layers[i], hidden_layers[i+1])
            )
        self.value_output_layer = torch.nn.Linear(hidden_layers[-1], 1)
        self.policy_output_layer = torch.nn.Linear(hidden_layers[-1], output_size)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        return self.policy_output_layer(x), self.value_output_layer(x)

    def softmax_policy(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        return action

    def greedy_policy(self, state):
        logits = self(state).detach().numpy()
        return np.argmax(logits)
    
    def reset(self):
        self.apply(FCPV.reset_weights)

    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Env(mp.Process):
    def __init__(self, id, create_env, pipe_end) -> None:
        super().__init__()
        self.id = id
        self.env = create_env()
        self.pipe_end = pipe_end

    def run(self):
        while True:
            cmd, args = self.pipe_end.recv()
            if cmd == 'reset':
                self.pipe_end.send(self.env.reset(args))
            elif cmd == 'step':
                self.pipe_end.send(self.env.step(args))
            # elif cmd == '_past_limit':
            #     worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                self.env.close(**kwargs)
                del self.env
                self.pipe_end.close()
                break


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
        return [coordinator_end.recv() for coordinator_end, _ in self.pipes]

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(
            ('step', {'action':actions[rank]}), 
            rank) for rank in range(self.n_workers)]
        results = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            o, r, d, i = parent_end.recv()
            results.append((o, 
                            np.array(r, dtype=np.float), 
                            np.array(d, dtype=np.float), 
                            i))
        return [np.vstack(block) for block in np.array(results).T]    


class A2C:
    def __init__(self) -> None:
        pass

    def train(self, max_episodes, n_workers):
        results = {}
        results['rewards'] = torch.zeros((max_episodes))
        results['times'] = torch.zeros((max_episodes))

        pipes = [mp.Pipe() for _ in range(n_workers)]
        workers = []
        for i in range(n_workers):
            worker = self.Worker(i)
            workers.append(worker)

        [w.start() for w in workers]
        [w.join() for w in workers]

        return results

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    print(env.reset())