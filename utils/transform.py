import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix

from utils.const import gdr_mode
from utils.io import parse_device


def get_coord2d_map(width, height, cam_K=None, device=None):
    device = cam_K.device if cam_K is not None else parse_device(device)
    coor2d_x, coor2d_y = torch.meshgrid(torch.arange(float(width)), torch.arange(float(height)),
                                        indexing='xy')  # [H, W]
    coor2d = torch.stack([coor2d_x, coor2d_y, torch.ones_like(coor2d_x)], dim=0).to(device)  # [3(XY1), H, W]
    if cam_K is not None:
        coor2d = torch.linalg.solve(cam_K, coor2d.reshape(3, -1)).reshape(3, height, width)
        # solve(K, M) == K.inv() @ M
    return coor2d[:2]  # [2(XY), H, W]


def get_bbox2d_from_mask(mask):
    # mask: [..., H, W]
    w_mask = mask.any(dim=-2)  # [..., W]
    h_mask = mask.any(dim=-1)  # [..., H]
    x0 = w_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
    y0 = h_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
    w = w_mask.sum(dim=-1)  # [...]
    h = h_mask.sum(dim=-1)  # [...]
    return torch.stack([x0 + w * .5, y0 + h * .5, w, h], dim=-1)  # [..., 4(XYWH)]


def calc_bbox2d_crop(bbox):
    crop_size, _ = bbox[:, 2:].max(dim=-1)  # [N]
    if gdr_mode:
        crop_size *= 1.5
    pad_size = int((crop_size.max() * .5).ceil())
    x0, y0 = (bbox[:, :2].T - crop_size * .5).round().int() + pad_size
    crop_size = crop_size.round().int()
    return crop_size, pad_size, x0, y0


def t_to_t_site(t, bbox, r=None, K=None):
    if K is not None:
        # if bbox is in image space
        t_site = t @ K.T  # [N, 3], (ox * tz, oy * tz, tz) = t.T @ K.T
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
        # if bbox is in image space
        t = torch.linalg.solve(K, t.T).T  # [N, 3], (inv(K) @ t).T
    return t


def rot_allo2ego(translation):
    """
    Compute rotation between ray to object centroid and optical center ray

    rot_ego = rot_allo2ego @ rot_allo

    :param translation: [..., 3]
    :return: [..., 3, 3]
    """
    cam_ray = torch.tensor([0., 0., 1.], device=translation.device)  # [3]
    obj_ray = F.normalize(translation, dim=-1)  # [..., 3]

    half_cos_theta = obj_ray[..., -1:] * .5  # [..., 1] \in [-.5, .5], cam_ray.dot(obj_ray), assume cam_ray (0., 0., 1.)
    cos_half_theta = (.5 + half_cos_theta).sqrt()  # [..., 1] \in [0., 1.]
    sin_half_theta = (.5 - half_cos_theta).sqrt()  # [..., 1] \in [0., 1.]

    # Compute rotation between ray to object centroid and optical center ray
    axis = F.normalize(torch.cross(cam_ray.expand_as(obj_ray), obj_ray), dim=-1)  # [..., 3]
    # Build quaternion representing the rotation around the computed axis
    quat_allo_to_ego = torch.cat([cos_half_theta, axis * sin_half_theta], dim=-1)  # [..., 4]
    return quaternion_to_matrix(quat_allo_to_ego)  # [..., 3, 3]
