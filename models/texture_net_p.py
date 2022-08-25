import torch
from torch import nn


def ConvBN(in_channels: int, out_channels: int , bias: bool, activation: str = None):
    layers = [
        nn.Conv2d(in_channels, out_channels, 1, bias=bias),
    ]
    if not bias:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class TextureNetP(nn.Module):
    def __init__(self, in_channels: int = 3+3, out_channels: int = 3, n_layers: int = 2, hidden_size: int = 128,
                 positional_encoding: list = None, positional_encoding_in_channels: int = None,
                 use_cosine_positional_encoding: bool = True):
        super().__init__()
        self.positional_encoding: list = positional_encoding
        self.positional_encoding_in_channels: int = positional_encoding_in_channels
        self.use_cosine_positional_encoding: bool = use_cosine_positional_encoding
        if self.positional_encoding is not None:
            if self.positional_encoding_in_channels is not None:
                self.conv1 = ConvBN(in_channels, self.positional_encoding_in_channels, False, None)
                in_channels = self.positional_encoding_in_channels
            in_channels *= 1 + len(self.positional_encoding) * (1 + self.use_cosine_positional_encoding)
        if n_layers > 1:
            layers = [ConvBN(in_channels, hidden_size, False, 'relu')] \
                     + [ConvBN(hidden_size, hidden_size, False, 'relu') for _ in range(n_layers - 2)] \
                     + [ConvBN(hidden_size, out_channels, True, None)]
        else:
            layers = [ConvBN(in_channels, out_channels, True, None)]
        self.layers = nn.Sequential(*layers)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x \in [-1, 1]
        if self.positional_encoding is not None:
            if self.positional_encoding_in_channels is not None:
                x = self.conv1(x)
            x_pos = x * torch.pi
            x_pos = torch.cat([x_pos * i for i in self.positional_encoding], dim=-3)
            if self.use_cosine_positional_encoding:
                x = torch.cat([x, x_pos.sin(), x_pos.cos()], dim=-3)
            else:
                x = torch.cat([x, x_pos.sin()], dim=-3)
        x = self.layers(x)
        x = self.act(x)
        return x
