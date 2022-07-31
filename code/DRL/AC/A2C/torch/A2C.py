import time
from tqdm import tqdm
import torch
import numpy as np


from .EnvironmentCoordinator import EnvironmentCoordinator
from code.DRL.nn.torch.FullyConnectedPolicyAction import FCPA
from code.DRL.nn.torch.FullyConnectedValue import FCV
from code.DRL.tools.Buffer import Buffer

class A2C:
    def __init__(self, create_env_fn,
                 actor_kwargs, critic_kwargs,
                 n_workers, n_step_max=10, gamma=0.99):
        self.envs = EnvironmentCoordinator(create_env_fn, n_workers)
        env = create_env_fn()
        self.state_space  = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.buffer = Buffer()
        self.policy_nn = FCPA(self.state_space, self.action_space, **actor_kwargs)
        self.value_nn = FCV(self.state_space, **critic_kwargs)
        
        self.gamma = gamma
        self.n_workers = n_workers
        self.n_step_max = n_step_max

    def train(self, max_episodes):
        results = {}
        results['rewards'] = torch.zeros((max_episodes))
        results['times'] = torch.zeros((max_episodes))
        start_time = time.time()
        t = 1

        for i in range(max_episodes):
            sum_rewards = 0
            t_start = t
            states = self.envs.reset()
            while True:
                actions = self.policy_nn.softmax_policy(states)
                states_p, rewards, dones, is_terminals = self.envs.step(actions)
                self.buffer.store(states, actions, rewards)
                sum_rewards += np.mean(rewards)

                if np.sum(dones) or t - t_start == self.n_step_max:
                    states_buffered, actions_buffered, rewards_buffered = self.buffer.get()
                    returns = self.get_returns(states_p, rewards_buffered, is_terminals)

                    self.optimize(states_buffered, actions_buffered, returns)
                    self.buffer.clear()
                    t_start = t

                if np.sum(dones): break
                t += 1
                states = states_p
                
            results['rewards'][i] = sum_rewards
            results['times'][i] = time.time() - start_time
    
            print("Episode:", i, "- Avg Reward: %d" % sum_rewards)

        return results

    def optimize(self, states, actions, returns):
        T = len(states)            
        discounts = torch.tensor(
            np.logspace(0, T, num=T, base=self.gamma, endpoint=False)).unsqueeze(-1)
        returns = torch.tensor(returns, dtype=torch.float).unsqueeze(-1)
        actions = torch.tensor(actions)

        logits = self.policy_nn(states)
        values = self.value_nn(states)

        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()

        td_errors = returns - values

        p_loss = -torch.mean(discounts * logpas * td_errors.detach().squeeze(-1))
        entropy_loss = -self.policy_nn.entropy_loss_weight * torch.mean(entropies)
        policy_loss = p_loss + entropy_loss
        self.policy_nn.train(policy_loss)

        value_loss = torch.mean(1/2 * td_errors**2)
        self.value_nn.train(value_loss)


    def get_returns(self, states_p, rewards, is_terminals):
        values_p = self.value_nn(states_p)
        R = values_p.detach().squeeze().numpy() * (1-np.array(is_terminals))
        T = len(rewards)
        returns = np.zeros((T, self.n_workers))
        for i in range(T):
            R = rewards[T-1-i] + self.gamma * R
            returns[T-1-i] = R

        return returns
        
    def reset(self):
        self.value_nn.reset()
        self.policy_nn.reset()

    def iterate(self, n_runs, max_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_runs, max_episodes))
        results['smoothed_rewards'] = torch.zeros((n_runs, max_episodes))
        results['times'] = torch.zeros((n_runs, max_episodes))
        results['smoothed_times'] = torch.zeros((n_runs, max_episodes))

        for i in tqdm(range(n_runs)):
            self.reset()
            res = self.train(max_episodes)

            results['rewards'][i] = res["rewards"]
            results['smoothed_rewards'][i] = torch.tensor([
                torch.mean(res["rewards"][max(i-99,0):i+1]) for i in range(max_episodes)])
            results['times'][i] = res["times"]
            results['smoothed_times'][i] = torch.tensor([
                torch.mean(res["times"][max(i-9,0):i+1]) for i in range(max_episodes)])

        self.envs.close()
        return results
