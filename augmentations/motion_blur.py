import torch
import torch.nn.functional as F

import config.const as cc


def draw_line(width: int, height: int, a: float, b: float, c: float, dtype: torch.dtype = cc.dtype) -> torch.Tensor:
    """
    Xiaolin Wu's line algorithm, a * x + b * y + c = 0
    https://www.rosettacode.org/wiki/Xiaolin_Wu%27s_line_algorithm#Python
    :param width: int
    :param height: int
    :param a: float
    :param b: float
    :param c: float
    :param dtype:
    :return: [H, W], CPU tensor
    """
    slope = - a / b
    if abs(slope) > 1:
        return draw_line(height, width, b, a, c, dtype).T
    img = torch.zeros(height + 1, width, dtype=dtype)
    intercept = - c / b
    x = torch.arange(width)
    y = slope * x + intercept
    for xi, yi, fi, mi in zip(x, y.floor().int(), y.frac(), y >= 0.):
        img[yi, xi] = (1. - fi) * mi
        img[yi + 1, xi] = fi * mi
    return img[:-1]


def _create_motion_kernel(angle: float = 0., kernel_size: int = 3, dtype: torch.dtype = cc.dtype, device=cc.device):
    """
    https://stackoverflow.com/questions/15579729/motion-blur-convolution-matrix-given-an-angle-and-magnitude
    https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
    :param kernel_size: int, odd numbers recommended
    :param angle: float \in [0., torch.pi)
    :return: [kernel_size, kernel_size]
    """
    angle = torch.tensor(angle) % torch.pi
    if angle <= torch.pi * .25 or angle >= torch.pi * .75:
        a = - torch.tan(angle)
        b = 1.
    else:
        a = 1.
        b = - 1. / torch.tan(angle)
    c = -.5 * (kernel_size - 1) * (a + b)
    kernel = draw_line(kernel_size, kernel_size, a, b, c, dtype).to(device)
    kernel = kernel[kernel.sum(dim=1).bool()][:, kernel.sum(dim=0).bool()]
    kernel /= kernel_size  # div kernel.sum()
    return kernel


def motion_blur(img: torch.Tensor, angle: float, kernel_size: int) -> torch.Tensor:
    """

    :param img: [N, C, H, W]
    :param angle: float \in [0, torch.pi)
    :param kernel_size: int, odd numbers recommended
    :return: [N, C, H, W]
    """
    kernel = _create_motion_kernel(angle, kernel_size, img.dtype, img.device)
    C = img.size(-3)
    return F.conv2d(img, kernel.expand(C, 1, -1, -1), padding='same', groups=C)
