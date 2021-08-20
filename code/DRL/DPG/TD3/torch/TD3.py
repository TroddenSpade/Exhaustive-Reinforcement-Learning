import os
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from code.DRL.mem.ReplayBuffer import ReplayBuffer
from code.DRL.nn.torch.FullyConnectedDeterministicPolicy import FCDP
from code.DRL.nn.torch.FullyConnectedQ import FCQ
from code.DRL.noise import NormalActionNoise


class TD3:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs,
                 update_actor_step,
                 gamma=0.99, tau=0.001, batch_size=64,
                 noise=0.1, noise_=0.2, clip_=0.5) -> None:
        self.t = 0
        self.frames = []
        self.gamma = gamma
        self.tau = tau
        self.update_actor_step = update_actor_step

        self.env = create_env_fn()
        n_states, n_actions = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.action_maxs = self.env.action_space.high
        self.action_mins = self.env.action_space.low
        self.tensor_maxs = torch.tensor(self.env.action_space.high)
        self.tensor_mins = torch.tensor(self.env.action_space.low)
        
        self.noise = NormalActionNoise((self.action_maxs+self.action_mins)/2, noise)
        self.noise_ = lambda : torch.normal((self.tensor_maxs+self.tensor_mins)/2, noise_)
        self.clip_ = clip_
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_shape=self.env.observation_space.shape,
                                          action_space=n_actions, batch_size=batch_size)

        self.actor_online = FCDP(input_size=n_states, output_size=n_actions,
                                        action_maxs=self.action_maxs, **actor_kwargs)
        self.actor_target = FCDP(input_size=n_states, output_size=n_actions,
                                        action_maxs=self.action_maxs, **actor_kwargs)
        self.critic1_online = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.critic1_target = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.critic2_online = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.critic2_target = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)


    def initialize(self):
        self.t = 0
        self.actor_online.reset()
        self.actor_target.reset()
        self.critic1_online.reset()
        self.critic1_target.reset()
        self.critic2_online.reset()
        self.critic2_target.reset()

        self.replay_buffer.clear()
        self.update_networks(reset=True)


    def update_networks(self, reset=False):
        tau = 1.0 if reset else self.tau
        for target, online in zip(self.critic1_target.parameters(),
                                  self.critic1_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(self.critic2_target.parameters(),
                                  self.critic2_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(self.actor_target.parameters(), 
                                  self.actor_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        
    def noisy_policy(self, state):
        self.actor_online.eval()
        mu = self.actor_online(state).detach().cpu().numpy()
        noisy_mu = mu + self.noise()
        clipped_mu = np.clip(noisy_mu, self.action_mins, self.action_maxs)
        return clipped_mu

    def greedy_policy(self, state):
        self.actor_online.eval()
        mu = self.actor_online(state).detach().cpu().numpy()
        return mu

    
    def play(self):
        frames = []
        state = self.env.reset()
        while True:
            frames.append(Image.fromarray(self.env.render(mode='rgb_array')))
            action = self.greedy_policy(state)
            state_p, _, done, _ = self.env.step(action)
            
            if done: break
            state = state_p
        self.frames.append(frames)

    def create_gifs(self, dir):
        for i, frames in enumerate(self.frames):
            path = os.path.join(dir, str(i)+".gif")
            with open(path, 'wb') as f:
                im = Image.new('RGB', frames[0].size)
                im.save(f, save_all=True, append_images=frames)


    def train(self, n_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_episodes))
        results['times'] = torch.zeros((n_episodes))
        start_time = time.time()

        for i in range(n_episodes):
            sum_rewards = 0
            state = self.env.reset()
            while True:
                self.t += 1
                action = self.noisy_policy(state)
                state_p, reward, done, info = self.env.step(action)
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_terminal = done and not is_truncated
                self.replay_buffer.store(state, action, reward, state_p, float(is_terminal))
                sum_rewards += reward

                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, states_p, is_terminals = self.replay_buffer.sample()
                    self.optimize_critic(states, actions, rewards, states_p, is_terminals)
                    if self.t % self.update_actor_step == 0:
                        self.optimize_actor(states)
                        self.update_networks()

                if done: break
                state = state_p

            results['rewards'][i] = sum_rewards
            results['times'][i] = time.time() - start_time
            print("Episode:", i,
                  "- Reward: %d" % sum_rewards, 
                  "- Avg Reward: %d" % torch.mean(results["rewards"][max(i-99,0):i+1]))

        return results


    def optimize_critic(self, states, actions, rewards, states_p, is_terminals):
        rewards = torch.tensor(rewards).unsqueeze(dim=1)
        is_terminals = torch.tensor(is_terminals).unsqueeze(dim=1)

        with torch.no_grad():
            self.actor_target.eval()
            target_actions_p = self.actor_target(states_p)
            target_actions_p = target_actions_p + \
                torch.clip(self.noise_(), -self.clip_, self.clip_)
            target_actions_p = torch.max(
                torch.min(target_actions_p, self.tensor_maxs), self.tensor_mins)

            self.critic1_target.eval()
            q1_sa_p = self.critic1_target(states_p, target_actions_p)
            self.critic2_target.eval()
            q2_sa_p = self.critic2_target(states_p, target_actions_p)

            q_sa_p = torch.min(q1_sa_p, q2_sa_p)
            q_target = rewards + self.gamma * q_sa_p * (1 - is_terminals)
            
        self.critic1_online.eval()
        q1_sa = self.critic1_online(states, actions)
        self.critic2_online.eval()
        q2_sa = self.critic2_online(states, actions)

        q1_loss = 0.5*torch.mean((q1_sa - q_target).pow(2))
        q2_loss = 0.5*torch.mean((q2_sa - q_target).pow(2))
        q_loss = q1_loss + q2_loss

        self.critic1_online.train()
        self.critic2_online.train()
        self.critic1_online.optimizer.zero_grad()
        self.critic2_online.optimizer.zero_grad()
        q_loss.backward()
        self.critic1_online.optimizer.step()
        self.critic2_online.optimizer.step()


    def optimize_actor(self, states):
        self.actor_online.eval()
        online_mu = self.actor_online(states)
        self.critic1_online.eval()
        q_s_mu = self.critic1_online(states, online_mu)
        policy_loss = -torch.mean(q_s_mu)

        self.actor_online.train()
        self.actor_online.optimizer.zero_grad()
        policy_loss.backward()
        self.actor_online.optimizer.step()


    def iterate(self, n_runs, n_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_runs, n_episodes))
        results['times'] = torch.zeros((n_runs, n_episodes))
        results['frames'] = []

        for i in tqdm(range(n_runs)):
            self.initialize()
            res = self.train(n_episodes)
            self.play()

            results['rewards'][i] = res["rewards"]
            results['times'][i] = res["times"]

        self.env.close()
        return results