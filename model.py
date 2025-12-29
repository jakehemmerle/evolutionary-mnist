import torch.nn as nn


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
    return MNISTConvNet(
        channels=params.cnn_channels,
        kernel_size=params.cnn_kernel_size,
        dropout=params.cnn_dropout,
        fc_hidden=params.cnn_fc_hidden,
    )
