import torch

#
class FCDP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers,
                action_max,
                activation_fn=torch.nn.functional.relu,
                out_activation_fn=torch.tanh,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=float('inf')) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.activation_fn = activation_fn
        self.out_activation_fn = out_activation_fn
        self.action_max = action_max

        self.hidden_norms = []
        self.hidden_layers = torch.nn.ModuleList()
        prev_size = input_size
        for layer_size in hidden_layers:
            self.hidden_layers.append(torch.nn.Linear(prev_size, layer_size))
            self.hidden_norms.append(torch.nn.LayerNorm(layer_size))
            prev_size = layer_size
        self.output_layer = torch.nn.Linear(layer_size, output_size)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)


    def format_(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        return states

    
    def forward(self, states):
        x = self.format_(states)

        for hidden_layer, norm_layer in zip(self.hidden_layers, self.hidden_norms):
            x = self.activation_fn(norm_layer(hidden_layer(x)))
        x = self.out_activation_fn(self.output_layer(x))
        return x * self.action_max
        
    
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
        self.apply(FCDP.reset_weights)

                    
if __name__ == "__main__":
    fcdp = FCDP(input_size=8, output_size=2, hidden_layers=(100,200,300),
                activation_fn=torch.nn.functional.relu,
                out_activation_fn=torch.tanh,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=float('inf'))
    print(fcdp)