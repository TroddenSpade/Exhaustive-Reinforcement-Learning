import torch

#
class FCQ(torch.nn.Module):
    '''
    states -> ...states_hidden_layers... 🡖 
                                           ...shared_hidden_layers... -> q
                                 actions 🡕 
    '''
    
    def __init__(self, states_input_size, actions_input_size, 
                states_hidden_layers, shared_hidden_layers,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=float("inf")) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.activation_fn = activation_fn

        self.states_hidden_norms = []
        self.states_hidden_layers = torch.nn.ModuleList()
        prev_size = states_input_size
        for layer_size in states_hidden_layers:
            self.states_hidden_layers.append(torch.nn.Linear(prev_size, layer_size))
            self.states_hidden_norms.append(torch.nn.LayerNorm(layer_size))
            prev_size = layer_size
        
        self.shared_hidden_norms = []
        self.shared_hidden_layers = torch.nn.ModuleList()
        prev_size = prev_size + actions_input_size
        for layer_size in shared_hidden_layers:
            self.shared_hidden_layers.append(torch.nn.Linear(prev_size, layer_size))
            self.shared_hidden_norms.append(torch.nn.LayerNorm(layer_size))
            prev_size = layer_size

        self.output_layer = torch.nn.Linear(prev_size, 1)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)


    def format_(self, states, actions):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        return states, actions


    def forward(self, states, actions):
        x, actions = self.format_(states, actions)

        for hidden_layer, norm_layer in zip(self.states_hidden_layers, self.states_hidden_norms):
            x = self.activation_fn(norm_layer(hidden_layer(x)))

        x = torch.cat((x, actions), dim=1)

        for hidden_layer, norm_layer in zip(self.shared_hidden_layers, self.shared_hidden_norms):
            x = self.activation_fn(norm_layer(hidden_layer(x)))
            
        return self.output_layer(x)


    def optimize(self, loss):
        self.train()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.grad_max_norm)
        self.optimizer.step()


    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reset(self):
        self.apply(FCQ.reset_weights)