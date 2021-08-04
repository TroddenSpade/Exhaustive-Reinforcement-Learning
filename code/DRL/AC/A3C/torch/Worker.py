import time
import torch
import numpy as np
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, id, results, global_T, create_env_fn, fcpa, fcv, buffer,
                global_policy_nn, global_value_nn, 
                n_step_max, T_max, gamma=0.99):
        super().__init__()
        self.id = id
        self.env = create_env_fn()
        # self.n_actions = self.env.action_space.n
        
        self.buffer = buffer
        self.policy_nn = fcpa
        self.value_nn = fcv
        self.policy_nn.load_state_dict(global_policy_nn.state_dict())
        self.value_nn.load_state_dict(global_value_nn.state_dict())

        self.global_policy_nn = global_policy_nn
        self.global_value_nn = global_value_nn

        self.results = results
        self.T = global_T
        self.T_MAX = T_max
        self.n_step_max = n_step_max
        self.gamma= gamma
        
    def get_returns(self, rewards, state_p, is_terminal):
        n = len(rewards)
        R = 0 if is_terminal else self.value_nn(state_p)
        returns = np.zeros(n)
        for i in range(n):
            R = rewards[n-1-i] + self.gamma * R
            returns[n-1-i] = R
        return returns

    def optimize(self, states, returns, actions):
        T = len(returns)
        discounts = torch.tensor(
            np.logspace(0, T, num=T, base=self.gamma, endpoint=False)).unsqueeze(dim=1)
        returns = torch.tensor(returns, dtype=torch.float).unsqueeze(dim=1)
        actions = torch.tensor(actions)

        values = self.value_nn(states)
        logits = self.policy_nn(states)

        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()

        td_errors = returns - values

        ###### policy optimization ######
        p_loss = -torch.mean(discounts * logpas * td_errors.detach())
        entropy_loss = -self.policy_nn.entropy_loss_weight * entropies.mean()
        policy_loss = p_loss +  entropy_loss

        self.global_policy_nn.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_nn.parameters(), self.policy_nn.grad_max_norm)
        for param, global_param in zip(self.policy_nn.parameters(),
                                    self.global_policy_nn.parameters()):
            if global_param.grad is None:
                global_param._grad = param.grad
        self.global_policy_nn.optimizer.step()
        self.policy_nn.load_state_dict(self.global_policy_nn.state_dict())

        ###### value optimization ######
        value_loss = torch.mean(1/2 * td_errors**2)
        
        self.global_value_nn.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_nn.parameters(), self.value_nn.grad_max_norm)
        for param, global_param in zip(self.value_nn.parameters(),
                                    self.global_value_nn.parameters()):
            if global_param.grad is None:
                global_param._grad = param.grad
        self.global_value_nn.optimizer.step()
        self.value_nn.load_state_dict(self.global_value_nn.state_dict())

    def run(self):
        start_time = time.time()
        t = 1
        while True:
            sum_rewards = 0
            t_start = t
            state = self.env.reset()
            while True:
                action = self.policy_nn.softmax_policy(state).item()
                state_p, reward, done, info = self.env.step(action)
                sum_rewards += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_terminal = done and not is_truncated
                self.buffer.store(state, action, reward)

                if done or t - t_start == self.n_step_max:
                    states, actions, rewards = self.buffer.get()
                    returns = self.get_returns(rewards, state_p, is_terminal)
                    self.optimize(states, returns, actions)
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
