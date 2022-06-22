import torch
from torch import nn


def ConvBN(in_channels: int, out_channels: int , bias: bool, activation: str = None):
    layers = [
        nn.Conv2d(in_channels, out_channels, 1, bias=bias),
        nn.BatchNorm2d(out_channels),
    ]
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class TextureNetP(nn.Module):
    def __init__(self, in_channels: int = 3+3, out_channels: int = 3, n_layers: int = 2, hidden_size: int = 128):
        super().__init__()
        if n_layers > 1:
            layers = [ConvBN(in_channels, hidden_size, True, 'relu')] \
                     + [ConvBN(hidden_size, hidden_size, True, 'relu') for _ in range(n_layers - 2)] \
                     + [ConvBN(hidden_size, out_channels, True, None)]
        else:
            layers = [ConvBN(in_channels, out_channels, True, None)]
        self.layers = nn.Sequential(*layers)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = self.act(x)
        return x
