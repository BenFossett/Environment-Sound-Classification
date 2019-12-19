from typing import Union, NamedTuple
import torch
from torch import nn
from torch.nn import functional as F

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: int, mode: str):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(p=dropout)

        # Layer 1 - 32 kernels with (3x3) receptive field, batch normalisation
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        # Layer 2 - 32 kernels with (3x3) receptive field, batch normalisation
        # followed by (2x2) max-pooling
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        # Layer 3 - 64 kernels with (3x3) receptive field, batch normalisation
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv3)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        # Layer 4 - 64 kernels with (3x3) receptive field, batch normalisation
        # followed by (2x2) max-pooling
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv4)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        # Layer 5 - Fully connected layer with 1024 hidden units
        if mode == "MLMC":
            self.fc1 = nn.Linear(26048, 1024)
        else:
            self.fc1 = nn.Linear(15488, 1024)

        # Output layer - ten units
        self.out = nn.Linear(1024, 10)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(specs)))
        x = F.relu(self.bn2(self.conv2(self.dropout(x))))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(self.dropout(x))))
        x = self.pool4(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.out(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
