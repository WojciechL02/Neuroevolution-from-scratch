import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, n_neurons) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, n_neurons))
        self.layers.append(nn.LeakyReLU())

        for _ in range(hidden_layers-1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Linear(n_neurons, output_size))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


