from tqdm import tqdm
import torch
import torch.multiprocessing as mp

from .Worker import Worker
from code.DRL.nn.torch.FullyConnectedPolicyAction import FCPA
from code.DRL.nn.torch.FullyConnectedValue import FCV
from code.DRL.mem.Buffer import Buffer

class A3C:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs, shared_optimizer,
                 n_workers, n_step_max=50, gamma=0.99):
        self.create_env_fn = create_env_fn
        env = create_env_fn()
        self.state_space  = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.actor_kwargs = actor_kwargs.copy()
        self.critic_kwargs = critic_kwargs.copy()
        actor_kwargs['optimizer'] = shared_optimizer
        critic_kwargs['optimizer'] = shared_optimizer

        self.global_policy_nn = FCPA(
            self.state_space, self.action_space, **actor_kwargs).share_memory()
        self.global_value_nn = FCV(
            self.state_space, **critic_kwargs).share_memory()

        self.gamma = gamma
        self.n_workers = n_workers
        self.n_step_max = n_step_max
        self.global_T = mp.Value('i', 0)


    def train(self, max_episodes):
        results = {}
        results['rewards'] = torch.zeros((max_episodes))
        results['times'] = torch.zeros((max_episodes))
        workers = []
        for i in range(self.n_workers):
            worker = Worker(i, results, self.global_T, self.create_env_fn, 
                FCPA(self.state_space, self.action_space, **self.actor_kwargs), 
                FCV(self.state_space, **self.critic_kwargs), 
                Buffer(),
                self.global_policy_nn, self.global_value_nn,
                self.n_step_max, max_episodes, self.gamma)
            workers.append(worker)

        [w.start() for w in workers]
        [w.join() for w in workers]

        return results


    def reset(self):
        self.global_T = mp.Value('i', 0)
        self.global_policy_nn.reset()
        self.global_value_nn.reset()


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
        return results
