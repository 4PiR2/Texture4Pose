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


def adapt_background_mean(img: torch.Tensor, mask: torch.Tensor, blend_saturation: float = 1.,
                          blend_light: float = 1.) -> torch.Tensor:
    """

    :param img: [N, 3(RGB), H, W] \in [0, 1]
    :param mask: [N, 1, H, W]
    :param blend_saturation: float, 0. -> keep original, 1. -> fully adapted
    :param blend_light: float, 0. -> keep original, 1. -> fully adapted
    :return: [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = rgb2hsl_torch(img)  # [N, 3(HSL), H, W]
    mask_count = mask.sum(dim=[-2, -1])  # [N, 1]
    bg_mean = (img_hsl * ~mask).sum(dim=[-2, -1]) / (H * W - mask_count)  # [N, 3(HSL)]
    obj_mean = (img_hsl * mask).sum(dim=[-2, -1]) / mask_count  # [N, 3(HSL)]
    mean_diff = bg_mean - obj_mean
    mean_diff[:, 0] = 0.
    mean_diff[:, 1] *= blend_saturation
    mean_diff[:, 2] *= blend_light
    img_hsl_obj_adapted = ((img_hsl + mean_diff[..., None, None]) * mask + img_hsl * ~mask).clamp(min=0., max=1.)
    # mean values might not be the same after clamping
    return hsl2rgb_torch(img_hsl_obj_adapted)


def adapt_background_histogram(img: torch.Tensor, mask: torch.Tensor, blend_saturation: float = 1.,
                               blend_light: float = 1.) -> torch.Tensor:
    """
    make the object's histogram the same as background's
    :param img: [N, 3(RGB), H, W] \in [0, 1]
    :param mask: [N, 1, H, W]
    :param blend_saturation: float, 0. -> keep original, 1. -> fully adapted
    :param blend_light: float, 0. -> keep original, 1. -> fully adapted
    :return: [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = rgb2hsl_torch(img)  # [N, 3(HSL), H, W]

    def per_channel(img_channel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        :param img_channel: [H, W]
        :param mask: [H, W]
        :return: [H, W]
        """
        obj_sort = img_channel[mask].argsort().argsort()
        cdf_bg, _ = img_channel[~mask].sort()
        img_channel = img_channel.clone()
        img_channel[mask] = cdf_bg[(obj_sort * (len(cdf_bg) / len(obj_sort))).long()]
        return img_channel

    def per_image(img_hsl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        :param img_hsl: [3(HSL), H, W]
        :param mask: [1, H, W]
        :return: [3(HSL), H, W]
        """
        if blend_saturation:
            img_hsl[1] = per_channel(img_hsl[1], mask[0]) * blend_saturation + img_hsl[1] * (1. - blend_saturation)
        if blend_light:
            img_hsl[2] = per_channel(img_hsl[2], mask[0]) * blend_light + img_hsl[2] * (1. - blend_light)
        return img_hsl

    return hsl2rgb_torch(torch.stack([per_image(i, m) for i, m in zip(img_hsl, mask)], dim=0))


def iso_noise(img: torch.Tensor, color_shift: float = .05, intensity: float = .5) -> torch.Tensor:
    """
    Apply poisson noise to image to simulate camera sensor noise.
    Args:
        img: [N, 3(RGB), H, W] \in [0, 1]
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
    Returns:
        [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = rgb2hsl_torch(img)
    stddev = img_hsl.std(dim=[-2, -1], unbiased=False)[:, 2, None, None]  # [N, 1, 1]
    luminance_noise = torch.poisson((stddev * intensity * 255.).expand(-1, H, W)) / 255.  # [N, H, W]
    img_hsl[:, 2] = (img_hsl[:, 2] + luminance_noise * (1. - img_hsl[:, 2])).clamp(min=0., max=1.)
    hue_color_noise = torch.randn_like(img_hsl[:, 2]) * color_shift * intensity  # [N, H, W]
    img_hsl[:, 0] = (img_hsl[:, 0] + hue_color_noise) % 1.
    return hsl2rgb_torch(img_hsl)


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
