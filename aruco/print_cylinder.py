import torch

import utils.print_a4


def unroll_cylinder_side(r: float, h: float = None, dpi: int = 72):
    if h is None:
        h = r * 2.
    H = utils.print_a4.meter2px(2. * torch.pi * r, dpi)
    W = utils.print_a4.meter2px(h, dpi)
    coord_3d = torch.empty(3, H, W)
    normal = torch.empty_like(coord_3d)
    theta = torch.linspace(0., 2. * torch.pi, H)[..., None]
    normal[0] = theta.cos()
    normal[1] = theta.sin()
    normal[2] = 0.
    coord_3d[:2] = normal[:2] * r
    coord_3d[2] = torch.linspace(-.5 * h, .5 * h, W)
    margin = 10
    img = torch.zeros(3, H + margin, W + 2)
    img[:, :-margin, 1:-1] = coord_3d / (2. * r) + .5
    img[:, -margin:] = img[:, -margin-1:-margin]
    img[:, -margin:, 0] = img[:, -margin:, -1] = 1.
    return img
