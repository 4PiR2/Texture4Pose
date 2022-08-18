# https://github.com/limacv/RGB_HSV_HSL
"""
# Pure Pytorch implementation of RGB to HSV/HSL conversion

This repository implement pytorch Tensor color space
conversion from rgb to hsv/hsl and backwards.

## Notes
1. The conversion process is differentiable in a natural way,
but since the mapping from RGB to HSV/HSL space is not
continuous in some place, it may not be a good idea to
perform back propagation.

2. The code was written to accept only image like tensors
`tensor.shape == [B x 3 x H x W]`, but it's easy to modify
the code to accept other shapes.

3. Reference is in [here](https://www.rapidtables.com/convert/color/index.html)

## Usage

example usage can be find in `test.py`.
Before using the function, please make sure that:
1. The input/output rbg/hsv/hsl tensors should all be normalized to 0~1 for each channel
2. The rgb format is RGB instead of BGR
2. The shape of the tensor match the requirement
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
    return torch.cat([hsv_h, hsl_s, hsl_l], dim=-3)


def hsl2hsv(hsl) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl.split(split_size=1, dim=-3)
    hsv_v = hsl_l + hsl_s * torch.minimum(hsl_l, 1. - hsl_l)
    hsv_s = torch.where(hsv_v == 0., torch.zeros_like(hsv_v), 2. * (1. - hsl_l / (hsv_v + 1e-12)))
    return torch.cat([hsl_h, hsv_s, hsv_v], dim=-3)


def rgb2hsl(rgb: torch.Tensor) -> torch.Tensor:
    return hsv2hsl(rgb2hsv(rgb))
    # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    # v_max, v_max_idx = rgb.max(dim=-3, keepdim=True)
    # v_min, _ = rgb.min(dim=-3, keepdim=True)
    # delta = v_max - v_min
    # rgb_r, rgb_g, rgb_b = rgb.split(split_size=1, dim=-3)
    # hsl_h = torch.empty_like(rgb_r)
    # mask_gt_0 = delta > 0.
    # mask_r, mask_g, mask_b = (v_max_idx == 0) & mask_gt_0, (v_max_idx == 1) & mask_gt_0, (v_max_idx == 2) & mask_gt_0
    # hsl_h[mask_r] = ((rgb_g[mask_r] - rgb_b[mask_r]) / delta[mask_r] % 6.)
    # hsl_h[mask_g] = ((rgb_b[mask_g] - rgb_r[mask_g]) / delta[mask_g] + 2.)
    # hsl_h[mask_b] = ((rgb_r[mask_b] - rgb_g[mask_b]) / delta[mask_b] + 4.)
    # hsl_h[~mask_gt_0] = 0.
    # hsl_h /= 6.
    # hsl_l_2 = v_max + v_min
    # hsl_l = hsl_l_2 * .5
    # hsl_s = torch.empty_like(hsl_l)
    # hsl_s[hsl_l_2 <= 0.] = 0.
    # hsl_s[hsl_l_2 >= 2.] = 0.
    # hsl_l_l_0_5 = (0. < hsl_l_2) & (hsl_l_2 < 1.)
    # hsl_l_g_0_5 = (1. <= hsl_l_2) & (hsl_l_2 < 2.)
    # hsl_s[hsl_l_l_0_5] = delta[hsl_l_l_0_5] / hsl_l_2[hsl_l_l_0_5]
    # hsl_s[hsl_l_g_0_5] = delta[hsl_l_g_0_5] / (2. - hsl_l_2[hsl_l_g_0_5])
    # return torch.cat([hsl_h, hsl_s, hsl_l], dim=-3)


def hsl2rgb(hsl: torch.Tensor) -> torch.Tensor:
    return hsv2rgb(hsl2hsv(hsl))
    # https://www.rapidtables.com/convert/color/hsl-to-rgb.html
    # hsl_h, hsl_s, hsl_l = hsl.split(split_size=1, dim=-3)
    # _c = (1. - (hsl_l * 2. - 1.).abs()) * hsl_s
    # _x = _c * (1. - (hsl_h * 6. % 2. - 1.).abs())
    # _m = hsl_l - _c * .5
    # idx = ((hsl_h * 6.).type(torch.uint8) % 6).expand(*([-1] * (hsl.ndim - 3)), 3, -1, -1)
    # rgb = torch.empty_like(hsl)
    # _o = torch.zeros_like(_c)
    # rgb[idx == 0] = torch.cat([_c, _x, _o], dim=-3)[idx == 0]
    # rgb[idx == 1] = torch.cat([_x, _c, _o], dim=-3)[idx == 1]
    # rgb[idx == 2] = torch.cat([_o, _c, _x], dim=-3)[idx == 2]
    # rgb[idx == 3] = torch.cat([_o, _x, _c], dim=-3)[idx == 3]
    # rgb[idx == 4] = torch.cat([_x, _o, _c], dim=-3)[idx == 4]
    # rgb[idx == 5] = torch.cat([_c, _o, _x], dim=-3)[idx == 5]
    # rgb += _m
    # return rgb


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
#     import cv2
#     import matplotlib.pyplot as plt
#     from torchvision.transforms import ToTensor
#
#     im_size = 16
#     for r in torch.linspace(0., 1., 5 + 1):
#         im = random_color_v_eq_1(im_size ** 2, r).reshape(im_size, im_size, 3)
#         plt.imshow(im)
#         plt.title(f'{r}')
#         plt.show()
#
#     img = cv2.imread('data/BOP/lm/test/000001/rgb/000001.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = ToTensor()(img).unsqueeze(0)
#     assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3, 'tensor shape should be like B x 3 x H x W'
#     hsv_tensor = rgb2hsv(img_tensor)
#     hsvback = hsv2rgb(hsv_tensor)
#     hsl_tensor = rgb2hsl(img_tensor)
#     hslback = hsl2rgb(hsl_tensor)
#
#     fig, axes = plt.subplots(1, 3)
#     axes[0].imshow(img_tensor[0].permute(1, 2, 0).numpy())
#     axes[1].imshow(hsvback[0].permute(1, 2, 0).numpy())
#     axes[2].imshow(hslback[0].permute(1, 2, 0).numpy())
#     plt.show()
