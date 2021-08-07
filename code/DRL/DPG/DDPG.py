import numpy as np

from code.DRL.mem.ReplayBuffer import ReplayBuffer
from code.DRL.nn.torch.FullyConnectedDeterministicPolicy import FCDP
from code.DRL.nn.torch.FullyConnectedQ import FCQ
from code.DRL.tools.OrnsteinUhlenbeckActionNoise import OUNoise

class DDPG:
    def __init__(self, create_env_fn, 
                 actor_kwargs, critic_kwargs, 
                 replay_buffer_max_size, 
                 gamma, tau) -> None:
        self.env = create_env_fn()
        n_states, n_actions = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(state_shape=self.env.observation_space.shape,
                                          action_space=n_actions ,max_size=replay_buffer_max_size)

        self.actor_online = FCDP(input_shape=n_states, output_shape=n_actions, *actor_kwargs)
        self.actor_target = FCDP(input_shape=n_states, output_shape=n_actions, *actor_kwargs)
        self.critic_online = FCQ(input_shape=n_states+n_actions, *critic_kwargs)
        self.crtic_target = FCQ(input_shape=n_states+n_actions, *critic_kwargs)

        self.noise = OUNoise(n_actions)

    def train(self):
        pass

    def optimize(self):
        pass