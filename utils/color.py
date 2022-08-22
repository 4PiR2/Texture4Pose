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
    # vFt._rgb2hsv

    r, g, b = rgb.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(rgb, dim=-3).values
    minc = torch.min(rgb, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc + 1e-12)
    # to prevent underflow during back-propagation
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr + 1e-12)
    # to prevent underflow during back-propagation, e.g., 2.4606e-39, which causes nan when calculating -0. / (x ** 2)

    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)

    return torch.stack((h, s, maxc), dim=-3)


def hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    return vFt._hsv2rgb(hsv)


def hsv2hsl(hsv: torch.Tensor) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/HSL_and_HSV#Interconversion
    hsv_h, hsv_s, hsv_v = hsv.split(split_size=1, dim=-3)
    hsl_l = hsv_v * (1. - hsv_s * .5)
    hsl_s = torch.where((hsl_l <= 0.) | (hsl_l >= 1.), torch.zeros_like(hsl_l),
                        (hsv_v - hsl_l) / (torch.minimum(hsl_l, 1. - hsl_l) + 1e-12))  # to prevent nan
    return torch.cat([hsv_h, hsl_s, hsl_l], dim=-3).clamp(min=0., max=1.)


def hsl2hsv(hsl: torch.Tensor) -> torch.Tensor:
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
