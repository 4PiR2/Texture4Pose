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


def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def random_color_v_eq_1(N: int, max_saturation: float = 1.) -> torch.Tensor:
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    # r = R * sqrt(random())
    # theta = random() * 2 * PI
    # x = centerX + r * cos(theta)
    # y = centerY + r * sin(theta)

    hsv = torch.rand(N, 3)
    hsv[:, 1] = hsv[:, 1].sqrt() * max_saturation
    hsv[:, 2] = 1.
    return hsv2rgb_torch(hsv[..., None, None])[..., 0, 0]  # [N, 3]


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
#     hsv_tensor = rgb2hsv_torch(img_tensor)
#     hsvback = hsv2rgb_torch(hsv_tensor)
#     hsl_tensor = rgb2hsl_torch(img_tensor)
#     hslback = hsl2rgb_torch(hsl_tensor)
#
#     fig, axes = plt.subplots(1, 3)
#     axes[0].imshow(img_tensor[0].permute(1, 2, 0).numpy())
#     axes[1].imshow(hsvback[0].permute(1, 2, 0).numpy())
#     axes[2].imshow(hslback[0].permute(1, 2, 0).numpy())
#     plt.show()
