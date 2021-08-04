import torch

## Fully-Connected Value Network
class FCV(torch.nn.Module):
    def __init__(self, input_shape, hidden_layers,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005,
                grad_max_norm=float("inf")) -> None:
        super().__init__()
        self.grad_max_norm = grad_max_norm
        self.activation_fn = activation_fn

        self.input_layer = torch.nn.Linear(input_shape, hidden_layers[0])
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = torch.nn.Linear(hidden_layers[-1], 1)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        return self.output_layer(x)

    def train(self, loss):
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
        self.apply(FCV.reset_weights)