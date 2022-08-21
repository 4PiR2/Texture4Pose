# https://github.com/limacv/RGB_HSV_HSL
"""
The code was written to accept only image like tensors `tensor.shape == [B x 3 x H x W]`.
The input/output rbg/hsv/hsl tensors should all be normalized to 0~1 for each channel.
"""

import torch
import torchvision.transforms.functional_tensor as vFt


def rgb2gray(rgb: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    return vFt.rgb_to_grayscale(rgb, num_output_channels)


def rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:
    return vFt._rgb2hsv(rgb)


def hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    return vFt._hsv2rgb(hsv)


def hsv2hsl(hsv) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/HSL_and_HSV#Interconversion
    hsv_h, hsv_s, hsv_v = hsv.split(split_size=1, dim=-3)
    hsl_l = hsv_v * (1. - hsv_s * .5)
    hsl_s = torch.where((hsl_l <= 0.) | (hsl_l >= 1.), torch.zeros_like(hsl_l),
                        (hsv_v - hsl_l) / (torch.minimum(hsl_l, 1. - hsl_l) + 1e-12))  # to prevent nan
    return torch.cat([hsv_h, hsl_s, hsl_l], dim=-3).clamp(min=0., max=1.)


def hsl2hsv(hsl) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl.split(split_size=1, dim=-3)
    hsv_v = hsl_l + hsl_s * torch.minimum(hsl_l, 1. - hsl_l)
    hsv_s = torch.where(hsv_v <= 0., torch.zeros_like(hsv_v), (1. - hsl_l / (hsv_v + 1e-12)) * 2.)  # to prevent nan
    return torch.cat([hsl_h, hsv_s, hsv_v], dim=-3).clamp(min=0., max=1.)


def rgb2hsl(rgb: torch.Tensor) -> torch.Tensor:
    return hsv2hsl(rgb2hsv(rgb))


def hsl2rgb(hsl: torch.Tensor) -> torch.Tensor:
    return hsv2rgb(hsl2hsv(hsl))


def random_color_v_eq_1(N: int, max_saturation: float = 1.) -> torch.Tensor:
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    # r = R * sqrt(random())
    # theta = random() * 2 * PI
    # x = centerX + r * cos(theta)
    # y = centerY + r * sin(theta)

    hsv = torch.rand(N, 3)
    hsv[:, 1] = hsv[:, 1].sqrt() * max_saturation
    hsv[:, 2] = 1.
    return hsv2rgb(hsv[..., None, None])[..., 0, 0]  # [N, 3]


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     im_size = 16
#     for r in torch.linspace(0., 1., 5 + 1):
#         im = random_color_v_eq_1(im_size ** 2, r).reshape(im_size, im_size, 3)
#         plt.imshow(im)
#         plt.title(f'{r}')
#         plt.show()
