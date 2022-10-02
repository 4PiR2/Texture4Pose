import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import utils.weight_init


class UpSamplingBackbone(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 6, num_filters: int = 256, kernel_size: int = 3):
        super().__init__()

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                           output_padding=output_padding, bias=False))
        layers.append(nn.BatchNorm2d(num_filters))
        layers.append(nn.ReLU(inplace=True))
        for i in range(0, num_layers, 2):
            if i >= 1:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            for _ in range(2):
                layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(num_filters))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                utils.weight_init.normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                utils.weight_init.constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                utils.weight_init.normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                utils.weight_init.normal_init(m, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
