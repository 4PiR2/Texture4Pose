import torch
from torch import nn


class GeoHead(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 3, kernel_size: int = 1):
        super().__init__()
        assert kernel_size == 1 or kernel_size == 3, 'Only support kenerl 1 and 3'

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0 if kernel_size == 1 else 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
