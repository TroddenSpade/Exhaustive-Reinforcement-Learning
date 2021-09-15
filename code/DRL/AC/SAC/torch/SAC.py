import os
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from code.DRL.mem.ReplayBuffer import ReplayBuffer
from code.DRL.nn.torch.FullyConnectedGaussianPolicy import FCGP
from code.DRL.nn.torch.FullyConnectedQ import FCQ
from code.DRL.nn.torch.FullyConnectedValue import FCV


class SAC:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs, value_kwargs,
                 gamma=0.99, tau=0.001, 
                 batch_size=64, buffer_max_size=1000000) -> None:
        self.t = 0
        self.frames = []
        self.gamma = gamma
        self.tau = tau
        # self.update_actor_step = update_actor_step

        self.env = create_env_fn()
        n_states, n_actions = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.action_maxs = self.env.action_space.high
        self.action_mins = self.env.action_space.low
        self.tensor_maxs = torch.tensor(self.env.action_space.high)
        self.tensor_mins = torch.tensor(self.env.action_space.low)
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_shape=self.env.observation_space.shape,
                                          max_size=buffer_max_size,
                                          action_space=n_actions, batch_size=batch_size)

        self.actor = FCGP(input_size=n_states, action_maxs=self.action_maxs, 
                                    n_actions=n_actions, **actor_kwargs)
        self.critic1 = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.critic2 = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.value_online = FCV(input_size=n_states, **value_kwargs)
        self.value_target = FCV(input_size=n_states, **value_kwargs)
        

    def initialize(self):
        self.t = 0
        self.actor.reset()
        self.critic1.reset()
        self.critic2.reset()
        self.value_online.reset()
        self.value_target.reset()

        self.replay_buffer.clear()
        self.update_networks(reset=True)


    def update_networks(self, reset=False):
        tau = 1.0 if reset else self.tau
        for target, online in zip(self.value_target.parameters(),
                                  self.value_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        
    def select_action(self, state):
        self.actor.eval()
        action, _ = self.actor.sample_action(state, reparam=False)
        a = action.detach().cpu().numpy()
        return a

    # def greedy_policy(self, state):
    #     self.actor_online.eval()
    #     action, _ = self.actor.sample_action(state, reparam=False)
    #     a = action.detach().cpu().numpy()
    #     print(a)
    #     return

    
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
                action = self.select_action(state)
                state_p, reward, done, info = self.env.step(action)
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_terminal = done and not is_truncated
                self.replay_buffer.store(state, action, reward, state_p, float(is_terminal))
                sum_rewards += reward

                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, states_p, is_terminals = self.replay_buffer.sample()
                    self.optimize(states, actions, rewards, states_p, is_terminals)
                    self.update_networks()

                if done: break
                state = state_p

            results['rewards'][i] = sum_rewards
            results['times'][i] = time.time() - start_time
            print("Episode:", i,
                  "- Reward: %d" % sum_rewards, 
                  "- Avg Reward: %d" % torch.mean(results["rewards"][max(i-99,0):i+1]))

        return results


    def optimize(self, states, actions, rewards, states_p, is_terminals):
        rewards = torch.tensor(rewards).unsqueeze(dim=1)
        is_terminals = torch.tensor(is_terminals).unsqueeze(dim=1)

        # value update
        self.value_online.eval()
        values = self.value_online(states)
        self.actor.eval()
        new_actions, log_probs = self.actor.sample_action(states, reparam=False)
        self.critic1.eval()
        self.critic2.eval()
        q1_sa_new = self.critic1(states, new_actions)
        q2_sa_new = self.critic2(states, new_actions)
        q_sa_new = torch.min(q1_sa_new, q2_sa_new)
        values_target = q_sa_new - log_probs
        value_loss = 0.5 * torch.mean((values - values_target).pow(2))

        self.value_online.train()
        self.value_online.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_online.optimizer.step()

        # actor update
        new_reparam_actions, reaparam_log_probs = self.actor.sample_action(states, reparam=True)
        q1_sa_reparam = self.critic1(states, new_reparam_actions)
        q2_sa_reparam = self.critic2(states, new_reparam_actions)
        q_sa_raparam = torch.min(q1_sa_reparam, q2_sa_reparam)
        actor_loss = torch.mean(reaparam_log_probs - q_sa_raparam)

        self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # critics update
        self.value_online.eval()
        values_p = self.value_target(states)
        q_sa_target = rewards + self.gamma * values_p * (1-is_terminals)
        q1_sa = self.critic1(states, actions)
        q2_sa = self.critic2(states, actions)

        q1_loss = 0.5 * torch.mean((q_sa_target - q1_sa).pow(2))
        q2_loss = 0.5 * torch.mean((q_sa_target - q2_sa).pow(2))
        q_loss = q1_loss + q2_loss

        self.critic1.train()
        self.critic2.train()
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        

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