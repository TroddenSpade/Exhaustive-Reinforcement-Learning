import time
import torch
from tqdm import tqdm

from code.DRL.mem.ReplayBuffer import ReplayBuffer
from code.DRL.nn.torch.FullyConnectedDeterministicPolicy import FCDP
from code.DRL.nn.torch.FullyConnectedQ import FCQ
from code.DRL.tools.OrnsteinUhlenbeckActionNoise import OUNoise

class DDPG:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs,  
                 gamma=0.99, tau=0.001, min_buffer=64) -> None:
        self.gamma = gamma
        self.tau = tau

        self.env = create_env_fn()
        n_states, n_actions = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.min_buffer = min_buffer
        self.replay_buffer = ReplayBuffer(state_shape=self.env.observation_space.shape,
                                          action_space=n_actions)

        self.actor_online = FCDP(input_size=n_states, output_size=n_actions, **actor_kwargs)
        self.actor_target = FCDP(input_size=n_states, output_size=n_actions, **actor_kwargs)
        self.critic_online = FCQ(input_size=n_states+n_actions, **critic_kwargs)
        self.critic_target = FCQ(input_size=n_states+n_actions, **critic_kwargs)

        self.noise = OUNoise(n_actions)


    def reset(self):
        self.noise.reset()
        self.actor_online.reset()
        self.actor_target.reset()
        self.critic_online.reset()
        self.critic_target.reset()
        self.replay_buffer.clear()
        self.update_networks(reset=True)

    
    def update_networks(self, reset=False):
        tau = 1.0 if reset else self.tau
        for target, online in zip(self.critic_target.parameters(), 
                                  self.critic_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(self.actor_target.parameters(), 
                                  self.actor_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        
    def choose_noisy_action(self, state):
        mu = self.actor_online(state).detach().cpu().numpy()
        mu_noisy = mu + self.noise()
        return mu_noisy


    def train(self, n_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_episodes))
        results['times'] = torch.zeros((n_episodes))
        start_time = time.time()

        for i in range(n_episodes):
            sum_rewards = 0
            state = self.env.reset()
            while True:
                action = self.choose_noisy_action(state)
                state_p, reward, done, info = self.env.step(action)
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_terminal = done and not is_truncated
                self.replay_buffer.store(state, action, reward, state_p, float(is_terminal))
                sum_rewards += reward

                if len(self.replay_buffer) >= self.min_buffer:
                    batches = self.replay_buffer.sample()
                    self.optimize(*batches)
                    self.update_networks()

                if done: break

                state = state_p

            results['rewards'][i] = sum_rewards
            results['times'][i] = time.time() - start_time

            print("Episode:", i, "- Avg Reward: %d" % sum_rewards)

        return results


    def optimize(self, states, actions, rewards, states_p, is_terminals):
        rewards = torch.tensor(rewards).unsqueeze(dim=1)
        is_terminals = torch.tensor(is_terminals).unsqueeze(dim=1)

        target_actions_p = self.actor_target(states_p)
        q_sa_p = self.critic_target(
                                states_p, target_actions_p).detach()
        q_target = rewards + self.gamma * q_sa_p * (1 - is_terminals)
        q_sa = self.critic_online(states, actions)
        td_error = q_sa - q_target
        q_loss = torch.mean(0.5 * td_error**2)
        self.critic_online.train(q_loss)

        online_mu = self.actor_online(states)
        q_s_mu = self.critic_online(states, online_mu)
        policy_loss = -torch.mean(q_s_mu)
        self.actor_online.train(policy_loss)


    def iterate(self, n_runs, n_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_runs, n_episodes))
        results['smoothed_rewards'] = torch.zeros((n_runs, n_episodes))
        results['times'] = torch.zeros((n_runs, n_episodes))
        results['smoothed_times'] = torch.zeros((n_runs, n_episodes))

        for i in tqdm(range(n_runs)):
            self.reset()
            res = self.train(n_episodes)

            results['rewards'][i] = res["rewards"]
            results['smoothed_rewards'][i] = torch.tensor([
                torch.mean(res["rewards"][max(i-99,0):i+1]) for i in range(n_episodes)])
            results['times'][i] = res["times"]
            results['smoothed_times'][i] = torch.tensor([
                torch.mean(res["times"][max(i-9,0):i+1]) for i in range(n_episodes)])

        self.env.close()
        return results