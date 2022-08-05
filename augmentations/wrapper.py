import torch
from torch import nn
import torchvision.transforms as T

import augmentations.color_augmentation
import augmentations.debayer
import utils.image_2d


class GaussNoise(nn.Module):
    def __init__(self, sigma=(0., .1), p: float = 1.):
        super().__init__()
        self.p: float = p
        if isinstance(sigma, float):
            sigma = (sigma,)
        self.sigma_min: float = sigma[0]
        self.sigma_max: float = sigma[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = len(x)
        dtype = x.dtype
        device = x.device
        sigma = torch.rand(N, dtype=dtype, device=device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        sigma *= torch.rand(N, dtype=dtype, device=device) <= self.p
        return (x + sigma[..., None, None, None] * torch.randn_like(x)).clamp(min=0., max=1.)


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p: float = 1.):
        super().__init__()
        self.p: float = p
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.color_jitter(xi) if torch.rand(1) <= self.p else xi for xi in x], dim=0)


class GaussianBlur(nn.Module):
    def __init__(self, sigma=(1., 3.), p: float = 1.):
        super().__init__()
        self.p: float = p
        if isinstance(sigma, float):
            sigma = (sigma,)
        sigma_min: float = sigma[0]
        sigma_max: float = sigma[-1]
        self.gaussian_blur = T.GaussianBlur(kernel_size=int(3.5 * sigma_max) * 2 + 1,
                                            sigma=(sigma_min, sigma_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.gaussian_blur(xi).clamp(min=0., max=1.)
                            if torch.rand(1) <= self.p else xi for xi in x], dim=0)


class CoarseDropout(nn.Module):
    def __init__(self, num_holes, width, p: float = 1.):
        super().__init__()
        self.p = p
        self.fn = lambda x: utils.image_2d.coarse_dropout(x, num_holes, width, fill_value=0., inplace=False)

    def forward(self, x: torch.Tensor):
        return torch.stack([self.fn(xi) if torch.rand(1) <= self.p else xi for xi in x], dim=0)


class ISONoise(nn.Module):
    def __init__(self, color_shift: float = .05, intensity: float = .5, p: float = 1.):
        super().__init__()
        self.p: float = p
        self.color_shift: float = color_shift
        self.intensity: float = intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([augmentations.color_augmentation.iso_noise(xi, self.color_shift, self.intensity)
                          if torch.rand(1) <= self.p else xi for xi in x.split(1)], dim=0)


class Debayer(nn.Module):
    def __init__(self, permute_channel: bool = True, p: float = 1.):
        super().__init__()
        self.p: float = p
        self.permute_channel: bool = permute_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([augmentations.debayer.debayer_aug(xi, self.permute_channel)
                          if torch.rand(1) <= self.p else xi for xi in x.split(1)], dim=0)
