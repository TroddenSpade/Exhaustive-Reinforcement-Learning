import time
import pickle
from tqdm import tqdm

import gym
import torch
import numpy as np
from torch import tensor
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


## Fully-Connected Policy Action Network
class FCPA(torch.nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=1, entropy_loss_weight=0.001) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.entropy_loss_weight = entropy_loss_weight
        self.activation_fn = activation_fn

        self.input_layer = torch.nn.Linear(input_shape, hidden_layers[0])
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = torch.nn.Linear(hidden_layers[-1], output_shape)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        return self.output_layer(x)

    def softmax_policy(self, state):
        logits = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        return action

    def greedy_policy(self, state):
        tens_state = torch.tensor(state).unsqueeze(dim=0)
        logits = self(tens_state).detach().numpy()
        return np.argmax(logits)

    def reset(self):
        self.apply(FCPA.reset_weights)

    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

## Fully-Connected Value Network
class FCV(torch.nn.Module):
    def __init__(self, input_shape, hidden_layers,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=float("inf")) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.activation_fn = activation_fn

        self.input_layer = torch.nn.Linear(input_shape, hidden_layers[0])
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = torch.nn.Linear(hidden_layers[-1], 1)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        return self.output_layer(x)

    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reset(self):
        self.apply(FCV.reset_weights)

class Buffer:
    def __init__(self) -> None:
        self.states = []
        self.rewards = []
        self.actions = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def clear(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def get(self):
        return self.states, self.actions, self.rewards


def create_env():
    return gym.make("CartPole-v1")

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)

class Worker(mp.Process):
    def __init__(self, id, global_T, create_env, fcpa, fcv, buffer,
                global_policy_nn, global_value_nn, 
                shared_policy_optimizer, shared_value_optimizer, 
                results, n_step_max, T_max=3000, gamma=0.99):
        super().__init__()
        self.id = id
        self.env = create_env()
        # self.n_actions = self.env.action_space.n
        
        self.buffer = buffer
        self.policy_nn = fcpa
        self.value_nn = fcv
        self.policy_nn.load_state_dict(global_policy_nn.state_dict())
        self.value_nn.load_state_dict(global_value_nn.state_dict())

        self.global_policy_nn = global_policy_nn
        self.global_value_nn = global_value_nn
        self.shared_policy_optimizer = shared_policy_optimizer
        self.shared_value_optimizer = shared_value_optimizer

        self.results = results
        self.T = global_T
        self.T_MAX = T_max
        self.n_step_max = n_step_max
        self.gamma= gamma
        
    def get_returns(self, rewards, value_p):
        n = len(rewards)
        R = value_p
        returns = np.zeros(n)
        for i in range(n):
            R = rewards[n-1-i] + self.gamma * R
            returns[n-1-i] = R
        return returns

    def optimize(self, states, actions, returns):
        T = len(returns)
        discounts = torch.tensor(
            np.logspace(0, T, num=T, base=self.gamma, endpoint=False)).unsqueeze(dim=1)
        returns = torch.tensor(returns, dtype=torch.float).unsqueeze(dim=1)
        actions = torch.tensor(actions, dtype=torch.float)

        values = self.value_nn(states)
        logits = self.policy_nn(states)

        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()

        td_errors = returns - values

        ###### policy optimization
        p_loss = -torch.mean(discounts * logpas * td_errors.detach())
        entropy_loss = -self.policy_nn.entropy_loss_weight * entropies.mean()
        policy_loss = p_loss +  entropy_loss

        self.shared_policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_nn.parameters(), self.policy_nn.grad_max_norm)

        for param, global_param in zip(self.policy_nn.parameters(),
                                    self.global_policy_nn.parameters()):
            if global_param.grad is None:
                global_param._grad = param.grad

        self.shared_policy_optimizer.step()
        self.policy_nn.load_state_dict(self.global_policy_nn.state_dict())

        ###### value optimization
        value_loss = torch.mean(1/2 * td_errors**2)
        
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_nn.parameters(), self.value_nn.grad_max_norm)

        for param, global_param in zip(self.value_nn.parameters(),
                                    self.global_value_nn.parameters()):
            if global_param.grad is None:
                global_param._grad = param.grad
            
        self.shared_value_optimizer.step()
        self.value_nn.load_state_dict(self.global_value_nn.state_dict())


    def run(self):
        start_time = time.time()
        t = 1
        while True:
            sum_rewards = 0
            t_start = t
            state = self.env.reset()
            while True:
                action = self.policy_nn.softmax_policy(state)
                state_p, reward, done, info = self.env.step(action)
                sum_rewards += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.store(state, action, reward)

                if done or t - t_start == self.n_step_max:
                    states, actions, rewards = self.buffer.get()
                    if is_failure:
                        value_p = 0
                    else:
                        value_p = self.value_nn(state_p[np.newaxis])
                    returns = self.get_returns(rewards, value_p)

                    self.optimize(states, actions, returns)
                    self.buffer.clear()
                    t_start = t

                if done: break
                t += 1
                state = state_p

            with self.T.get_lock():
                if self.T.value >= self.T_MAX: break
                self.results['rewards'][self.T.value] = sum_rewards
                self.results['times'][self.T.value] = time.time() - start_time
                self.T.value += 1
                
            print("Worker:", self.id, "- Episode:", self.T.value, "- Reward: %d" % sum_rewards)


class A3C:
    def __init__(self, create_env, FCPA, FCV, Buffer, Worker,
                 n_step_max=50, policy_lr=0.0005, value_lr=0.0007):
        self.create_env = create_env
        env = create_env()
        self.state_space  = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.Worker = Worker
        self.FCPA = FCPA
        self.FCV = FCV
        self.Buffer = Buffer

        self.policy_lr = policy_lr
        self.value_lr = value_lr

        self.global_policy_nn = FCPA(self.state_space, self.action_space, (128,64)).share_memory()
        self.global_value_nn = FCV(self.state_space, (256,128)).share_memory()
        self.shared_policy_optimizer = SharedAdam(self.global_policy_nn.parameters(), lr=policy_lr)
        self.shared_value_optimizer = SharedAdam(self.global_value_nn.parameters(), lr=value_lr)

        self.n_step_max = n_step_max
        self.global_T = mp.Value('i', 0)


    def train(self, max_episodes, n_workers):
        results = {}
        results['rewards'] = torch.zeros((max_episodes))
        results['times'] = torch.zeros((max_episodes))
        workers = []
        for i in range(n_workers):
            worker = self.Worker(i, self.global_T, self.create_env, 
                                 self.FCPA(self.state_space, self.action_space, (128,64)), 
                                 self.FCV(self.state_space, (256,128)), 
                                 self.Buffer(),
                                 self.global_policy_nn, self.global_value_nn,
                                 self.shared_policy_optimizer, self.shared_value_optimizer,
                                 results, self.n_step_max, max_episodes)
            workers.append(worker)

        [w.start() for w in workers]
        [w.join() for w in workers]

        return results


    def reset(self):
        self.global_T = mp.Value('i', 0)
        self.global_policy_nn.reset()
        self.global_value_nn.reset()


    def iterate(self, n_runs, max_episodes, n_workers):
        results = {}
        results['rewards'] = torch.zeros((n_runs, max_episodes))
        results['smoothed_rewards'] = torch.zeros((n_runs, max_episodes))
        results['times'] = torch.zeros((n_runs, max_episodes))
        results['smoothed_times'] = torch.zeros((n_runs, max_episodes))

        for i in tqdm(range(n_runs)):
            self.reset()
            res = self.train(max_episodes, n_workers)

            results['rewards'][i] = res["rewards"]
            results['smoothed_rewards'][i] = torch.tensor([
                torch.mean(res["rewards"][max(i-99,0):i+1]) for i in range(max_episodes)])
            results['times'][i] = res["times"]
            results['smoothed_times'][i] = torch.tensor([
                torch.mean(res["times"][max(i-9,0):i+1]) for i in range(max_episodes)])
        return results


if __name__ == "__main__":
    agent = A3C(create_env, FCPA, FCV, Buffer, Worker, n_step_max=50)
    res = agent.iterate(n_runs=5, max_episodes=1000, n_workers=4)

    file = open("A3C_res.pkl", "wb")
    pickle.dump(res, file)
    file.close()

    # file = open("A3C_res.pkl",'br')
    # res = pickle.load(file)

    smoothed_times_mean = torch.mean(res['smoothed_times'], axis=0)
    smoothed_times_min = torch.min(res['smoothed_times'], axis=0)[0]
    smoothed_times_max = torch.max(res['smoothed_times'], axis=0)[0]

    smoothed_rewards_mean = torch.mean(res['smoothed_rewards'], axis=0)
    smoothed_rewards_min = torch.min(res['smoothed_rewards'], axis=0)[0]
    smoothed_rewards_max = torch.max(res['smoothed_rewards'], axis=0)[0]

    plt.figure(1)
    plt.plot(smoothed_times_min, 'y', linewidth=1)
    plt.plot(smoothed_times_max, 'y', linewidth=1)
    plt.plot(smoothed_times_mean, 'y', label='NFQ', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_times_mean)), 
        smoothed_times_min, smoothed_times_max, facecolor='y', alpha=0.2)

    plt.figure(2)
    plt.plot([0,1000],[500,500], '--', color='crimson')
    plt.plot(smoothed_rewards_min, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_max, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_mean, 'deepskyblue', label='NFQ', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_rewards_mean)), 
        smoothed_rewards_min, smoothed_rewards_max, facecolor='deepskyblue', alpha=0.2)

    plt.show()