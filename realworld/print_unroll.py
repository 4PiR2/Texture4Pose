import torch

from models.main_model import MainModel
import utils.image_2d
import utils.print_paper


def unroll_cylinder_strip(scale: float = .05, margin: float = .01, border: int = 1, dpi: int = 72, model: MainModel = None):
    """

    :param scale: float
    :param margin: float, rate
    :param border: int, px
    :param dpi: int
    :param model:
    :return:
    """
    r = 1.
    h = r * 2.
    H = utils.print_paper.meter2px(2. * torch.pi * r * scale, dpi)
    W = utils.print_paper.meter2px(h * scale, dpi)
    margin = int(torch.ceil(torch.tensor(H * margin)))
    coord_3d = torch.empty(3, H, W)
    normal = torch.empty_like(coord_3d)
    theta = torch.linspace(0., 2. * torch.pi, H)[..., None]
    normal[0] = theta.cos()
    normal[1] = theta.sin()
    normal[2] = 0.
    coord_3d[:2] = normal[:2] * r
    coord_3d[2] = torch.linspace(-.5 * h, .5 * h, W)
    img = torch.zeros(3, H + margin, W + border * 2)
    if model is not None:
        model.eval()
        with torch.no_grad():
            img[:, :H, border: border + W] = model.forward_texture(
                # texture_mode='xyz',
                coord_3d_normalized=coord_3d[None] * .5 + .5, normal=normal[None], mask=torch.ones(1, 1, H, W, dtype=torch.bool)
            )
    else:
        img[:, :H, border: border + W] = coord_3d * .5 + .5
    img[:, H:] = img[:, H - 1: H]
    img[:, H:, :border] = img[:, H:, border + W:] = 1.
    return img


def get_spectrum_info(y: torch.Tensor, freq_sample: float = None):
    n = y.shape[-1]
    if freq_sample is None:
        freq_sample = float(n)
    y_fft = torch.fft.rfft(y, dim=-1, norm='forward')
    amplitude = y_fft.abs()
    amplitude[..., 1:] *= 2.
    phase = y_fft.angle()
    freq = torch.arange(amplitude.shape[-1], dtype=amplitude.dtype, device=amplitude.device) * (freq_sample / n)
    return freq, amplitude, phase


def _unroll_canonical_sphericon(canonical_coord_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    :param [..., 2, H, W], canonical_coord_2d, center (0., 0.)
    :return: [..., C, H, W], unrolled sphericon of range (-1., 1.)
    """
    coord_3d = torch.zeros_like(torch.cat([canonical_coord_2d, canonical_coord_2d[..., :1, :, :]], dim=-3))
    normal = torch.zeros_like(coord_3d)
    alpha = torch.tensor(torch.pi / (2. * 2. ** .5), device=canonical_coord_2d.device)

    v2 = (torch.stack([1.5 * alpha.sin(), .5 * alpha.cos()]) * 2. ** .5)[..., None, None]
    vec_v2 = canonical_coord_2d - v2
    theta_v2 = (alpha - .5 * torch.pi - torch.atan2(vec_v2[..., 1, :, :], vec_v2[..., 0, :, :])) * 2. ** .5
    dist_v2 = torch.linalg.vector_norm(vec_v2, ord=2, dim=-3) / 2. ** .5
    x2 = dist_v2 * theta_v2.cos()
    nx2 = .5 * 2. ** .5 * theta_v2.cos()
    y2 = 1. - dist_v2
    ny2 = torch.full_like(y2, .5 * 2. ** .5)
    z2 = dist_v2 * theta_v2.sin()
    nz2 = .5 * 2. ** .5 * theta_v2.sin()
    m2 = (0. <= theta_v2) & (theta_v2 <= torch.pi) & (dist_v2 <= 1.)
    coord_3d += torch.stack([x2, y2, z2], dim=-3) * m2[..., None, :, :]
    normal += torch.stack([nx2, ny2, nz2], dim=-3) * m2[..., None, :, :]
    del vec_v2, theta_v2, dist_v2, x2, y2, z2, nx2, ny2, nz2

    v1 = (torch.stack([alpha.sin(), -alpha.cos()]) / 2. ** .5)[..., None, None]
    vec_v1 = canonical_coord_2d - v1
    theta_v1 = (alpha - .5 * torch.pi + torch.atan2(vec_v1[..., 1, :, :], vec_v1[..., 0, :, :])) * 2. ** .5
    dist_v1 = torch.linalg.vector_norm(vec_v1, ord=2, dim=-3) / 2. ** .5
    x1 = dist_v1 - 1.
    nx1 = torch.full_like(x1, -.5 * 2. ** .5)
    y1 = dist_v1 * theta_v1.cos()
    ny1 = .5 * 2. ** .5 * theta_v1.cos()
    z1 = -dist_v1 * theta_v1.sin()
    nz1 = -.5 * 2. ** .5 * theta_v1.sin()
    m1 = (0. <= theta_v1) & (theta_v1 <= torch.pi) & (dist_v1 <= 1.) & ~m2
    coord_3d += torch.stack([x1, y1, z1], dim=-3) * m1[..., None, :, :]
    normal += torch.stack([nx1, ny1, nz1], dim=-3) * m1[..., None, :, :]
    del vec_v1, theta_v1, dist_v1, x1, y1, z1, nx1, ny1, nz1

    v3 = -v1
    vec_v3 = canonical_coord_2d - v3
    theta_v3 = (alpha - .5 * torch.pi - torch.atan2(vec_v3[..., 1, :, :], vec_v3[..., 0, :, :])) * 2. ** .5
    dist_v3 = torch.linalg.vector_norm(vec_v3, ord=2, dim=-3) / 2. ** .5
    x3 = -dist_v3 * theta_v3.cos()
    nx3 = -.5 * 2. ** .5 * theta_v3.cos()
    y3 = dist_v3 - 1.
    ny3 = torch.full_like(y3, -.5 * 2. ** .5)
    z3 = dist_v3 * theta_v3.sin()
    nz3 = .5 * 2. ** .5 * theta_v3.sin()
    m3 = (0. <= theta_v3) & (theta_v3 <= torch.pi) & (dist_v3 <= 1.) & ~m2 & ~m1
    coord_3d += torch.stack([x3, y3, z3], dim=-3) * m3[..., None, :, :]
    normal += torch.stack([nx3, ny3, nz3], dim=-3) * m3[..., None, :, :]
    del vec_v3, theta_v3, dist_v3, x3, y3, z3, nx3, ny3, nz3, v1, v3

    v0 = -v2
    vec_v0 = canonical_coord_2d - v0
    theta_v0 = (alpha - .5 * torch.pi + torch.atan2(vec_v0[..., 1, :, :], vec_v0[..., 0, :, :])) * 2. ** .5
    dist_v0 = torch.linalg.vector_norm(vec_v0, ord=2, dim=-3) / 2. ** .5
    x0 = 1. - dist_v0
    nx0 = torch.full_like(x0, .5 * 2. ** .5)
    y0 = -dist_v0 * theta_v0.cos()
    ny0 = -.5 * 2. ** .5 * theta_v0.cos()
    z0 = -dist_v0 * theta_v0.sin()
    nz0 = -.5 * 2. ** .5 * theta_v0.sin()
    m0 = (0. <= theta_v0) & (theta_v0 <= torch.pi) & (dist_v0 <= 1.) & ~m2 & ~m1 & ~m3
    coord_3d += torch.stack([x0, y0, z0], dim=-3) * m0[..., None, :, :]
    normal += torch.stack([nx0, ny0, nz0], dim=-3) * m0[..., None, :, :]
    del vec_v0, theta_v0, dist_v0, x0, y0, z0, nx0, ny0, nz0, v2, v0

    mask = (m2 | m1 | m3 | m0)[..., None, :, :]

    return coord_3d, normal, mask


def unroll_sphericon(scale: float, theta: float = -.9, dpi: int = 72, model: MainModel = None):
    """

    :param scale: float
    :param theta: float, rotation
    :param dpi: int
    :return:
    """
    _alpha = torch.pi / (2. * 2. ** .5)
    _x0 = (1.5 - float(torch.tensor(2. * _alpha).cos())) * 2. ** .5 * scale
    _y0 = 2. ** .5 * scale
    _angle = torch.tensor(theta + _alpha - .5 * torch.pi)
    _box = torch.tensor([[_x0, _y0], [-_x0, _y0], [-_x0, -_y0], [_x0, -_y0]]) @ \
          torch.tensor([[_angle.cos(), -_angle.sin()], [_angle.sin(), _angle.cos()]]).T
    _wh = _box.max(dim=0)[0] - _box.min(dim=0)[0]
    w, h = float(_wh[0]), float(_wh[1])

    W = utils.print_paper.meter2px(w, dpi)
    H = utils.print_paper.meter2px(h, dpi)
    dtype = torch.float32
    device = 'cpu'
    coord_2d = torch.stack(torch.meshgrid(
        torch.linspace(-.5 * w, .5 * w, W, dtype=dtype),
        torch.linspace(.5 * h, -.5 * h, H, dtype=dtype),
        indexing='xy'), dim=0).to(device)  # [2(XY), H, W]

    theta = torch.tensor(-theta)
    transformation = torch.tensor([[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]]) / scale
    coord_2d = (transformation @ coord_2d.reshape(2, -1)).reshape(2, H, W)
    coord_3d, normal, mask = _unroll_canonical_sphericon(coord_2d)
    _y, _x = mask[0].nonzero().T
    _x_min, _x_max, _y_min, _y_max = _x.min(), _x.max() + 1, _y.min(), _y.max() + 1
    coord_3d = coord_3d[..., _y_min:_y_max, _x_min:_x_max]
    normal = normal[..., _y_min:_y_max, _x_min:_x_max]
    mask = mask[..., _y_min:_y_max, _x_min:_x_max]

    if model == 'cb':
        cb_num_cycles = 2
        img = ((coord_3d + 1.) * cb_num_cycles).int() % 2
    elif model is not None:
        model.eval()
        with torch.no_grad():
            img = model.forward_texture(
                # texture_mode='xyz',
                coord_3d_normalized=coord_3d[None] * .5 + .5, normal=normal[None], mask=mask[None]
            )[0]
    else:
        img = coord_3d * .5 + .5

    mask = mask.expand(3, -1, -1)
    img[~mask] = 1.

    border = 1
    if border > 0:
        img[..., :-border] *= mask[..., :-border] | ~mask[..., border:]
        img[..., border:] *= mask[..., border:] | ~mask[..., :-border]
        img[..., :-border, :] *= mask[..., :-border, :] | ~mask[..., border:, :]
        img[..., border:, :] *= mask[..., border:, :] | ~mask[..., :-border, :]

    return img
