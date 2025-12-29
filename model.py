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


class MNISTConvNet(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, fc_hidden: int):
        super().__init__()
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14
            nn.Conv2d(channels, channels * 2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14 -> 7
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((channels * 2) * 7 * 7, fc_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(fc_hidden, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(params) -> nn.Module:
    # Local import to avoid circular imports (training.py imports build_model)
    model_type = getattr(params, "model_type", "mlp")
    if model_type == "cnn":
        return MNISTConvNet(
            channels=getattr(params, "cnn_channels", 32),
            kernel_size=getattr(params, "cnn_kernel_size", 3),
            dropout=getattr(params, "cnn_dropout", 0.0),
            fc_hidden=getattr(params, "cnn_fc_hidden", 128),
        )
    return MNISTNet(
        hidden_size=getattr(params, "hidden_size", 128),
        num_layers=getattr(params, "num_layers", 2),
    )
