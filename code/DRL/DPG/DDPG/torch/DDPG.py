import os
import time
import torch
from tqdm import tqdm
from PIL import Image


from code.DRL.mem.ReplayBuffer import ReplayBuffer
from code.DRL.nn.torch.FullyConnectedDeterministicPolicy import FCDP
from code.DRL.nn.torch.FullyConnectedQ import FCQ
from code.DRL.noise import OrnsteinUhlenbeckActionNoise

class DDPG:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs,  
                 gamma=0.99, tau=0.001, batch_size=64) -> None:
        self.gamma = gamma
        self.tau = tau
        self.frames = []

        self.env = create_env_fn()
        n_states, n_actions = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        action_maxs = self.env.action_space.high
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_shape=self.env.observation_space.shape,
                                          action_space=n_actions, batch_size=batch_size)
        self.noise = OrnsteinUhlenbeckActionNoise(n_actions)

        self.actor_online = FCDP(input_size=n_states, output_size=n_actions,
                                        action_maxs=action_maxs, **actor_kwargs)
        self.actor_target = FCDP(input_size=n_states, output_size=n_actions,
                                        action_maxs=action_maxs, **actor_kwargs)
        self.critic_online = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)
        self.critic_target = FCQ(states_input_size=n_states, 
                                    actions_input_size=n_actions, **critic_kwargs)


    def initialize(self):
        self.noise.reset()
        self.replay_buffer.clear()

        self.actor_online.reset()
        self.actor_target.reset()
        self.critic_online.reset()
        self.critic_target.reset()

        self.update_networks(reset=True)

    
    def update_networks(self, reset=False):
        tau = 1.0 if reset else self.tau
        for target, online in zip(self.critic_target.parameters(), 
                                  self.critic_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(self.actor_target.parameters(), 
                                  self.actor_online.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        
    def noisy_policy(self, state):
        self.actor_online.eval()
        mu = self.actor_online(state).detach().cpu().numpy()
        mu_noisy = mu + self.noise()
        return mu_noisy

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
                action = self.noisy_policy(state)
                state_p, reward, done, info = self.env.step(action)
                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_terminal = done and not is_truncated
                self.replay_buffer.store(state, action, reward, state_p, float(is_terminal))
                sum_rewards += reward

                if len(self.replay_buffer) >= self.batch_size:
                    batches = self.replay_buffer.sample()
                    self.optimize(*batches)
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

        self.actor_target.eval()
        target_actions_p = self.actor_target(states_p)
        self.critic_target.eval()
        q_sa_p = self.critic_target(
                                states_p, target_actions_p).detach()
        q_target = rewards + self.gamma * q_sa_p * (1 - is_terminals)
        self.critic_online.eval()
        q_sa = self.critic_online(states, actions)
        td_error = q_sa - q_target
        q_loss = 0.5*torch.mean(td_error.pow(2))
        self.critic_online.optimize(q_loss)

        self.actor_online.eval()
        online_mu = self.actor_online(states)
        self.critic_online.eval()
        q_s_mu = self.critic_online(states, online_mu)
        policy_loss = -torch.mean(q_s_mu)
        self.actor_online.optimize(policy_loss)


    def iterate(self, n_runs, n_episodes):
        results = {}
        results['rewards'] = torch.zeros((n_runs, n_episodes))
        results['times'] = torch.zeros((n_runs, n_episodes))

        for i in tqdm(range(n_runs)):
            self.initialize()
            res = self.train(n_episodes)
            self.play()

            results['rewards'][i] = res["rewards"]
            results['times'][i] = res["times"]

        self.env.close()
        return results