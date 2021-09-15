import torch

## Fully-Connected Gaussian Policy Network
class FCGP(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, 
                n_actions, action_maxs,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=1, entropy_loss_weight=0.001,
                c_min=1e-6, c_max=1, noise=1e-6) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.entropy_loss_weight = entropy_loss_weight
        self.c_min = c_min
        self.c_max = c_max
        self.noise = noise
        self.activation_fn = activation_fn
        self.action_maxs = torch.tensor(action_maxs)

        self.hidden_layers = torch.nn.ModuleList()
        prev_size = input_size
        for layer_size in hidden_layers:
            self.hidden_layers.append(torch.nn.Linear(prev_size, layer_size))
            prev_size = layer_size
        self.mu = torch.nn.Linear(prev_size, n_actions)
        self.sigma = torch.nn.Linear(prev_size, n_actions)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.clip(sigma, min=self.c_min, max=self.c_max)
        return mu, sigma


    def sample_action(self, states, reparam=True):
        mu, sigma = self(states)
        dist = torch.distributions.Normal(mu, sigma)
        if reparam:
            sampled_actions = dist.rsample()
        else:
            sampled_actions = dist.sample()
        
        actions = torch.tanh(sampled_actions) * self.action_maxs
        log_probs = dist.log_prob(sampled_actions) - \
                    torch.log(1-actions.pow(2) + self.noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return actions, log_probs


    # def greedy_policy(self, state):
    #     logits = self(state).detach()
    #     action = torch.argmax(logits)
    #     return action.numpy()

    def reset(self):
        self.apply(FCGP.reset_weights)

    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()