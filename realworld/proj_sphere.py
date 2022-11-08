import torch

import config.const as cc


def equirectangular(w: int, h: int = None, lat_0: float = 0., lon_0: float = 0., dtype=cc.dtype) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Equirectangular_projection
    if h is None:
        h = w // 2
    assert 0 < 2 * h <= w
    lon, lat = torch.meshgrid(torch.linspace(-torch.pi, torch.pi, w, dtype=dtype) + lon_0,
                              torch.linspace(.5 * torch.pi, -.5 * torch.pi, h, dtype=dtype) + lat_0,
                              indexing='xy')
    nx = lon.cos() * lat.cos()
    ny = lon.sin() * lat.cos()
    nz = lat.sin()
    normal = torch.stack([nx, ny, nz], dim=0)
    return normal


def orthographic():
    # https://en.wikipedia.org/wiki/Orthographic_map_projection
    pass


# if __name__ == '__main__':
#     from utils.image_2d import visualize
#     visualize(equirectangular(500, 250) * .5 + .5)
#     a = 0
