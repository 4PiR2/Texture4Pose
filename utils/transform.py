import torch

from utils.io import parse_device


def get_coor2d(width, height, cam_K=None, device=None):
    device = cam_K.device if cam_K is not None else parse_device(device)
    coor2d_x, coor2d_y = torch.meshgrid(torch.arange(float(width)), torch.arange(float(height)),
                                        indexing='xy')  # [H, W]
    coor2d = torch.stack([coor2d_x, coor2d_y, torch.ones_like(coor2d_x)], dim=0).to(device)  # [3(XY1), H, W]
    if cam_K is not None:
        coor2d = torch.linalg.solve(cam_K, coor2d.reshape(3, -1)).reshape(3, height, width)
        # solve(K, M) == K.inv() @ M
    return coor2d[:2]  # [2(XY), H, W]


def calculate_bbox_crop(bbox):
    crop_size, _ = bbox[:, 2:].max(dim=-1)  # [N]
    pad_size = int((crop_size.max() * .5).ceil())
    x0, y0 = (bbox[:, :2].T - crop_size * .5).round().int() + pad_size
    crop_size = crop_size.round().int()
    return crop_size, pad_size, x0, y0


def t_to_t_site(t, bbox, r=None, K=None):
    if K is not None:
        t_site = t @ K.T  # [N, 3], (ox*tz, oy*tz, tz) = t.T @ K.T
    else:
        t_site = t.clone()
    t_site[:, :2] = (t_site[:, :2] / t_site[:, 2:] - bbox[:, :2]) / bbox[:, 2:]
    # (dx, dy, .) = ((ox - cx) / w, (oy - cy) / h, .)
    t_site[:, 2] /= r  # (., ., dz) = (., ., tz / r)
    return t_site


def t_site_to_t(t_site, bbox, r=None, K=None):
    t = t_site.clone()
    t[:, 2] *= r  # (., ., tz) = (., ., dz * r)
    t[:, :2] = t[:, :2] * bbox[:, 2:] + bbox[:, :2]  # (ox, oy, .) = (dx * w + cx, dy * h + cy, .)
    t[:, :2] *= t[:, 2:]  # (ox * tz, oy * tz, .)
    if K is not None:
        t = torch.linalg.solve(K, t.T).T  # [N, 3], (inv(K) @ t).T
    return t
