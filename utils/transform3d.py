from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix


def t_to_t_site(t: torch.Tensor, bbox: torch.Tensor, r: Union[torch.Tensor, float] = 1., cam_K: torch.Tensor = None) \
        -> torch.Tensor:
    """
    :param t: [N, 3]
    :param bbox: [N, 4(XYWH)]
    :param r: float, crop_size / max(W, H)
    :param cam_K: cam_K: [3, 3]
    :return: [N, 3]
    """
    if cam_K is not None:
        # if bbox is in image space
        cam_K /= float(cam_K[-1, -1])
        t_site = t @ cam_K.T  # [N, 3], (ox * tz, oy * tz, tz) = t.T @ K.T
    else:
        t_site = t.clone()
    t_site[:, :2] = (t_site[:, :2] / t_site[:, 2:] - bbox[:, :2]) / bbox[:, 2:]
    # (dx, dy, .) = ((ox - cx) / w, (oy - cy) / h, .)
    t_site[:, 2] /= r  # (., ., dz) = (., ., tz / r)
    return t_site


def t_site_to_t(t_site: torch.Tensor, bbox: torch.Tensor, r: Union[torch.Tensor, float] = 1.,
                cam_K: torch.Tensor = None) -> torch.Tensor:
    """
    :param t_site: [N, 3]
    :param bbox: [N, 4(XYWH)]
    :param r: float, crop_size / max(W, H)
    :param cam_K: [3, 3]
    :return: [N, 3]
    """
    t = t_site.clone()
    t[:, 2] *= r  # (., ., tz) = (., ., dz * r)
    t[:, :2] = t[:, :2] * bbox[:, 2:] + bbox[:, :2]  # (ox, oy, .) = (dx * w + cx, dy * h + cy, .)
    t[:, :2] *= t[:, 2:]  # (ox * tz, oy * tz, .)
    if cam_K is not None:
        # if bbox is in image space
        cam_K /= float(cam_K[-1, -1])
        t = torch.linalg.solve(cam_K, t.T).T  # [N, 3], (inv(K) @ t).T
    return t


def rot_allo2ego(translation: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation between ray to object centroid and optical center ray

    rot_ego = rot_allo2ego @ rot_allo

    :param translation: [..., 3]
    :return: [..., 3, 3]
    """
    cam_ray = torch.tensor([0., 0., 1.], dtype=translation.dtype, device=translation.device)  # [3]
    obj_ray = F.normalize(translation, dim=-1)  # [..., 3]

    half_cos_theta = obj_ray[..., -1:] * .5  # [..., 1] \in [-.5, .5], cam_ray.dot(obj_ray), assume cam_ray (0., 0., 1.)
    cos_half_theta = (.5 + half_cos_theta).sqrt()  # [..., 1] \in [0., 1.]
    sin_half_theta = (.5 - half_cos_theta).sqrt()  # [..., 1] \in [0., 1.]

    # Compute rotation between ray to object centroid and optical center ray
    axis = F.normalize(torch.cross(cam_ray.expand_as(obj_ray), obj_ray), dim=-1)  # [..., 3]
    # Build quaternion representing the rotation around the computed axis
    quat_allo_to_ego = torch.cat([cos_half_theta, axis * sin_half_theta], dim=-1)  # [..., 4]
    return quaternion_to_matrix(quat_allo_to_ego)  # [..., 3, 3]
