import torch
from torch import nn

from models.head.geo_head import GeoHead


class EPHead(nn.Module):
    def __init__(self, in_channels: int = 256, kernel_size: int = 1):
        super().__init__()
        self.layers = GeoHead(in_channels, 2, kernel_size)
        self.fc = nn.Linear(in_channels, 2, bias=True)

    def forward(self, x: torch.Tensor):
        N, _, H, W = x.shape
        w2d_raw = self.layers(x)
        log_weight_scale = self.fc(x.mean(dim=[-2, -1]))
        return w2d_raw, log_weight_scale
