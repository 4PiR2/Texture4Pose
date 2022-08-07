from typing import Callable, Union

import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

import augmentations.color_augmentation
import augmentations.debayer
import utils.image_2d


class _Base(nn.Module):
    def __init__(self, fn: Callable = lambda x, *_: x,
                 *values_range: Union[float, list[float], tuple[float], tuple[float, float]],
                 per_channel: float = 0., p: float = 1.):
        super().__init__()
        self.fn: Callable = fn
        self.p: float = p
        self.per_channel: float = per_channel
        self.values_range: list[Union[list[float], tuple[float], tuple[float, float]]] = \
            [(v,) if isinstance(v, (float, int)) else v for v in values_range]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device
        *N, C = x.shape[:-2]
        values = torch.stack([torch.lerp(
            torch.tensor(v[0], dtype=dtype, device=device),
            torch.tensor(v[-1], dtype=dtype, device=device),
            torch.rand(*N, C, dtype=dtype, device=device),
        ) for v in self.values_range], dim=0)
        per_image_mask = torch.rand(*N, dtype=dtype, device=device) > self.per_channel
        values[:, per_image_mask] = values[:, per_image_mask, :1]
        values *= torch.rand(*N, 1, dtype=dtype, device=device) <= self.p
        y = self.fn(x, *values[..., None, None])
        return y.clamp(min=0., max=1.)


class Add(_Base):
    def __init__(self, value: Union[float, list[float], tuple[float], tuple[float, float]],
                 per_channel: float = 0., p: float = 1.):
        super().__init__(lambda x, a: x + a, value, per_channel=per_channel, p=p)


class Mult(_Base):
    def __init__(self, value: Union[float, list[float], tuple[float], tuple[float, float]],
                 per_channel: float = 0., p: float = 1.):
        super().__init__(lambda x, a: x * a, value, per_channel=per_channel, p=p)


class GaussNoise(_Base):
    def __init__(self, sigma=(0., .1), p: float = 1.):
        super().__init__(lambda x, s: x + s * torch.randn_like(x), sigma, per_channel=0., p=p)


class ColorJitter(nn.Module):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p: float = 1.):
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


class Sharpen(nn.Module):
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://github.com/rasmushaugaard/surfemb/blob/53e1852433a3b2b84fedc7a3a01674fe1b6189cc/surfemb/data/tfms.py#L26
    # factor := strength + 1.
    def __init__(self, sharpness_factor=(1., 3.), p: float = 1.):
        super().__init__()
        self.p: float = p
        if isinstance(sharpness_factor, float):
            sharpness_factor = (sharpness_factor,)
        self.sharpness_min: float = sharpness_factor[0]
        self.sharpness_max: float = sharpness_factor[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factors = torch.rand([len(x)], dtype=x.dtype) * (self.sharpness_max - self.sharpness_min) + self.sharpness_min
        return torch.stack([F.adjust_sharpness(xi, factor)
                            if torch.rand(1) <= self.p else xi for xi, factor in zip(x, factors)], dim=0)


class CoarseDropout(nn.Module):
    def __init__(self, num_holes=10, width=8, p: float = 1.):
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


# if __name__ == '__main__':
#     aug = Add((-20., 20.), per_channel=.5, p=.5)
#     x = torch.rand(4, 5, 6, 7)
#     y = aug(x)
#     a = 0
