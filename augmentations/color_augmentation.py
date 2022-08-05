import torch

import utils.color


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
    img_hsl = utils.color.rgb2hsl_torch(img)  # [N, 3(HSL), H, W]
    mask_count = mask.sum(dim=[-2, -1])  # [N, 1]
    bg_mean = (img_hsl * ~mask).sum(dim=[-2, -1]) / (H * W - mask_count)  # [N, 3(HSL)]
    obj_mean = (img_hsl * mask).sum(dim=[-2, -1]) / mask_count  # [N, 3(HSL)]
    mean_diff = bg_mean - obj_mean
    mean_diff[:, 0] = 0.
    mean_diff[:, 1] *= blend_saturation
    mean_diff[:, 2] *= blend_light
    img_hsl_obj_adapted = ((img_hsl + mean_diff[..., None, None]) * mask + img_hsl * ~mask).clamp(min=0., max=1.)
    # mean values might not be the same after clamping
    return utils.color.hsl2rgb_torch(img_hsl_obj_adapted)


def adapt_background_histogram(img: torch.Tensor, mask: torch.Tensor, blend_saturation: float = 1.,
                               blend_light: float = 1., p: float = 1.) -> torch.Tensor:
    """
    make the object's histogram the same as background's
    :param img: [N, 3(RGB), H, W] \in [0, 1]
    :param mask: [N, 1, H, W]
    :param blend_saturation: float, 0. -> keep original, 1. -> fully adapted
    :param blend_light: float, 0. -> keep original, 1. -> fully adapted
    :param p: float
    :return: [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = utils.color.rgb2hsl_torch(img)  # [N, 3(HSL), H, W]

    def per_channel(img_channel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        https://en.wikipedia.org/wiki/Histogram_equalization
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
        if blend_saturation and torch.rand(1) <= p:
            img_hsl[1] = per_channel(img_hsl[1], mask[0]) * blend_saturation + img_hsl[1] * (1. - blend_saturation)
        if blend_light and torch.rand(1) <= p:
            img_hsl[2] = per_channel(img_hsl[2], mask[0]) * blend_light + img_hsl[2] * (1. - blend_light)
        return img_hsl

    return utils.color.hsl2rgb_torch(torch.stack([per_image(i, m) for i, m in zip(img_hsl, mask)], dim=0))


def iso_noise(img: torch.Tensor, color_shift: float = .05, intensity: float = .5) -> torch.Tensor:
    """
    Apply poisson noise to image to simulate camera sensor noise.
    https://github.com/albumentations-team/albumentations/blob/8e958a324cb35d3adf13c03b180b3dc066ef21d5/albumentations/augmentations/transforms.py#L1406
    Args:
        img: [N, 3(RGB), H, W] \in [0, 1]
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
    Returns:
        [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = utils.color.rgb2hsl_torch(img)
    stddev = img_hsl.std(dim=[-2, -1], unbiased=False)[:, 2, None, None]  # [N, 1, 1]
    luminance_noise = torch.poisson((stddev * intensity * 255.).expand(-1, H, W)) / 255.  # [N, H, W]
    img_hsl[:, 2] = (img_hsl[:, 2] + luminance_noise * (1. - img_hsl[:, 2])).clamp(min=0., max=1.)
    hue_color_noise = torch.randn_like(img_hsl[:, 2]) * color_shift * intensity  # [N, H, W]
    img_hsl[:, 0] = (img_hsl[:, 0] + hue_color_noise) % 1.
    return utils.color.hsl2rgb_torch(img_hsl)
