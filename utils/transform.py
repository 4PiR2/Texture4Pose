import torch

from utils.const import gdr_mode


def calculate_bbox_crop(bbox):
    crop_size, _ = bbox[:, 2:].max(dim=-1)  # [N]
    if gdr_mode:
        crop_size *= 1.5
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
