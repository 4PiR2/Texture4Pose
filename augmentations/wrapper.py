from typing import Callable, Union

import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as vF

import augmentations.color_augmentation
import augmentations.debayer
import augmentations.motion_blur
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
            torch.rand(*N, C if self.per_channel else 1, dtype=dtype, device=device),
        ) for v in self.values_range], dim=0) \
            if self.values_range else torch.empty(0, *N, C, dtype=dtype, device=device)
        if self.per_channel:
            per_image_mask = torch.rand(*N, dtype=dtype, device=device) > self.per_channel
            values[:, per_image_mask] = values[:, per_image_mask, :1]
        values *= torch.rand(*N, 1, dtype=dtype, device=device) <= self.p
        y = self.main(x, *values[..., None, None])
        return y.clamp(min=0., max=1.)

    def main(self, x: torch.Tensor, *values) -> torch.Tensor:
        return self.fn(x, *values)


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


class _BaseI(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_channel: float = 0.

    def main(self, x: torch.Tensor, *values) -> torch.Tensor:
        return torch.cat([
            self.fn(xi, *vi)
            if torch.rand(1) <= self.p else xi
            for xi, *vi in zip(x.split(1, dim=0), *[v[..., 0, 0, 0] for v in values])
        ], dim=0)


class ColorJitter(_BaseI):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p: float = 1.):
        super().__init__(T.ColorJitter(brightness, contrast, saturation, hue), p=p)


class GaussianBlur(_BaseI):
    def __init__(self, sigma=(1., 3.), p: float = 1.):
        if isinstance(sigma, (float, int)):
            sigma = (sigma,)
        super().__init__(T.GaussianBlur(kernel_size=int(3.5 * sigma[-1]) * 2 + 1, sigma=(sigma[0], sigma[-1])), p=p)


class CoarseDropout(_BaseI):
    def __init__(self, num_holes=10, width=8, p: float = 1.):
        super().__init__(lambda x: utils.image_2d.coarse_dropout(x, num_holes, width, fill_value=0., inplace=False),
                         p=p)


class ISONoise(_BaseI):
    def __init__(self, color_shift: float = .05, intensity: float = .5, p: float = 1.):
        super().__init__(lambda x: augmentations.color_augmentation.iso_noise(x, color_shift, intensity), p=p)


class Debayer(_BaseI):
    def __init__(self, permute_channel: bool = True, p: float = 1.):
        super().__init__(lambda x: augmentations.debayer.debayer_aug(x, permute_channel), p=p)


class Sharpen(_BaseI):
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://github.com/rasmushaugaard/surfemb/blob/53e1852433a3b2b84fedc7a3a01674fe1b6189cc/surfemb/data/tfms.py#L26
    # factor := strength + 1.
    def __init__(self, sharpness_factor=(1., 3.), p: float = 1.):
        super().__init__(vF.adjust_sharpness, sharpness_factor, p=p)


class MotionBlur(_BaseI):
    def __init__(self, kernel_size=(1., 5.), p: float = 1.):
        # parameter is the half kernel size to make sure the kernel size is an odd integer in the end
        angle = (0., torch.pi)
        super().__init__(lambda x, a, k: augmentations.motion_blur.motion_blur(x, angle=a, kernel_size=int(k) * 2 + 1),
                         angle, kernel_size, p=p)


# if __name__ == '__main__':
#     import utils.io
#     from utils.image_2d import visualize
#
#     im = utils.io.read_img_file('/data/coco/train2017/000000000009.jpg')
#     # aug = CoarseDropout(num_holes=10, width=8, p=.5)
#     # aug = Sharpen(sharpness_factor=(1., 3.), p=1.)
#     aug = MotionBlur((9., 9.), p=.5)
#     y = aug(im)
#     visualize(im)
#     visualize(y)
#     a = 0
