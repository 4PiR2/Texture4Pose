import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from utils.weight_init import normal_init, constant_init


class RotWithRegionHead(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1, num_regions=64):
        super().__init__()

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kenerl 2, 3 and 4"
        assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kenerl 1 and 3"
        pad = 0 if output_kernel_size == 1 else 1

        features = []
        features.append(
            nn.ConvTranspose2d(
                in_channels,
                num_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            )
        )
        features.append(nn.BatchNorm2d(num_filters))
        features.append(nn.ReLU(inplace=True))
        for i in range(num_layers):
            if i >= 1:
                features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            features.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            features.append(nn.BatchNorm2d(num_filters))
            features.append(nn.ReLU(inplace=True))

            features.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            features.append(nn.BatchNorm2d(num_filters))
            features.append(nn.ReLU(inplace=True))

        self.region_output_dim = num_regions + 1  # add one channel for bg

        features.append(
            nn.Conv2d(
                num_filters,
                1 + 3 + self.region_output_dim,
                kernel_size=output_kernel_size,
                padding=pad,
                bias=True,
            )
        )
        self.features = nn.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def forward(self, x):
        x = self.features(x)
        mask, coor_3d, region = x.split([1, 3, x.shape[1] - 4], dim=1)
        return mask, coor_3d, region
