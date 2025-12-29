import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        input_size = 28 * 28

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
