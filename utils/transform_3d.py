from typing import Union

import cv2
from matplotlib.axes import Axes
import numpy as np
import pytorch3d.transforms
import torch
import torch.nn.functional as F


def normalize_cam_K(cam_K: torch.Tensor) -> torch.Tensor:
    return cam_K / cam_K[..., -1:, -1:]


def t_to_t_site(t: torch.Tensor, bbox: torch.Tensor, r: Union[torch.Tensor, float] = 1., cam_K: torch.Tensor = None) \
        -> torch.Tensor:
    """
    :param t: [..., 3]
    :param bbox: [..., 4(XYWH)]
    :param r: [...], crop_size / max(W, H)
    :param cam_K: cam_K: [..., 3, 3]
    :return: [..., 3]
    """
    if cam_K is not None:
        # if bbox is in image space
        cam_K = normalize_cam_K(cam_K)
        t = (t[..., None, :] @ cam_K.transpose(-2, -1))[..., 0, :]  # [..., 3], (ox * tz, oy * tz, tz) = t.T @ K.T
    t_site = t.clone()
    t_site[..., :2] = (t[..., :2] / t[..., 2:] - bbox[..., :2]) / bbox[..., 2:]
    # (dx, dy, .) = ((ox - cx) / w, (oy - cy) / h, .)
    t_site[..., 2] /= r  # (., ., dz) = (., ., tz / r)
    return t_site


def t_site_to_t(t_site: torch.Tensor, bbox: torch.Tensor, r: Union[torch.Tensor, float] = 1.,
                cam_K: torch.Tensor = None) -> torch.Tensor:
    """
    :param t_site: [..., 3]
    :param bbox: [..., 4(XYWH)]
    :param r: [...], crop_size / max(W, H)
    :param cam_K: [..., 3, 3]
    :return: [..., 3]
    """
    t = t_site.clone()
    t[..., :2] = (t_site[..., :2] * bbox[..., 2:] + bbox[..., :2]) * t_site[..., 2:]
    # (ox * dz, oy * dz, dz) = ((dx * w + cx) * dz, (dy * h + cy) * dz, dz)
    t *= r[..., None]  # (ox * tz, oy * tz, tz)
    if cam_K is not None:
        # if bbox is in image space
        cam_K = normalize_cam_K(cam_K)
        t = torch.linalg.solve(cam_K, t[..., None])[..., 0]  # [..., 3], (inv(K) @ t).T
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
    return pytorch3d.transforms.quaternion_to_matrix(quat_allo_to_ego)  # [..., 3, 3]


def denormalize_coord_3d(normalized_coord_3d: torch.Tensor, obj_size: torch.Tensor) -> torch.Tensor:
    """

    :param normalized_coord_3d: [..., 3(XYZ), H, W] \in [0, 1]
    :param obj_size: [..., 3(XYZ)]
    :return: [..., 3(XYZ), H, W]
    """
    return (normalized_coord_3d - .5) * obj_size[..., None, None]


def normalize_coord_3d(coord_3d: torch.Tensor, obj_size: torch.Tensor) -> torch.Tensor:
    """

    :param coord_3d: [..., 3(XYZ), H, W]
    :param obj_size: [..., 3(XYZ)]
    :return: [..., 3(XYZ), H, W] \in [0, 1]
    """
    return coord_3d / obj_size[..., None, None] + .5


def solve_pnp(coord_3d: torch.Tensor, coord_2d: torch.Tensor, mask: torch.Tensor = None, ransac: bool = True) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    assume K = I

    :param coord_3d: [N, 3(XYZ), H, W]
    :param coord_2d: [N, 2(XY), H, W]
    :param mask: [N, 1, H, W] or None
    :param ransac: bool
    :return: [N, 3, 3], [N, 3]
    """
    dtype = coord_3d.dtype
    device = coord_3d.device
    N, _, H, W = coord_3d.shape
    pnp_cam_R_m2c = torch.empty(N, 3, 3, dtype=dtype, device=device)
    pnp_cam_t_m2c = torch.empty(N, 3, dtype=dtype, device=device)
    if mask is None:
        mask = torch.ones(N, 1, H, W, dtype=torch.bool, device=device)
    else:
        mask = mask.to(dtype=dtype).round().bool()
    if ransac:
        mask_inlier = torch.empty_like(mask)
    else:
        mask_inlier = mask
    for i in range(N):
        x = coord_3d[i].permute(1, 2, 0)[mask[i, 0]].detach().cpu().numpy()
        y = coord_2d[i].permute(1, 2, 0)[mask[i, 0]].detach().cpu().numpy()
        if ransac:
            _, pred_R_exp, pred_t, inliers = cv2.solvePnPRansac(x, y, np.eye(3), None, iterationsCount=10000,
                                                                reprojectionError=1e-3, flags=cv2.SOLVEPNP_ITERATIVE)
            inliers = inliers.flatten()
            m = mask[i, 0].detach().cpu().numpy()
            m_i, m_j = np.where(m)
            m_inlier = np.zeros_like(m)
            m_inlier[m_i[inliers], m_j[inliers]] = 2
            mask_inlier[i, 0] = torch.tensor(m_inlier)
        else:
            _, pred_R_exp, pred_t = cv2.solvePnP(x, y, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        pred_R, _ = cv2.Rodrigues(pred_R_exp)
        pnp_cam_R_m2c[i] = torch.tensor(pred_R)
        pnp_cam_t_m2c[i] = torch.tensor(pred_t.flatten())
    return pnp_cam_R_m2c, pnp_cam_t_m2c, mask_inlier


def show_pose(ax: Axes, cam_K: torch.Tensor, cam_R_m2c: torch.Tensor, cam_t_m2c: torch.Tensor, obj_size: torch.Tensor,
              bbox: torch.Tensor = None, bbox_zoom_out=1.5,  adjust_axis: bool = True, alpha: float = 1.) \
        -> Axes:
    """

    :param ax:
    :param cam_K: [3, 3]
    :param cam_R_m2c: [3, 3]
    :param cam_t_m2c: [3]
    :param obj_size: [3(XYZ)]
    :param bbox: [4(XYWH)]
    :param adjust_axis: bool
    :param alpha: float
    :return:
    """
    dtype = cam_R_m2c.dtype
    device = cam_R_m2c.device
    verts = torch.tensor([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
                          [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                          [0, 0, 0]], dtype=dtype, device=device)  # [8+1, 3]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
    vert_colors = ['#ffff00', '#00ff00', '#404040', '#ff0000', '#bfbfbf', '#00ffff', '#0000ff', '#ff00ff', '#000000']
    faces = [[0, 1, 2, 3], [0, 3, 7, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [4, 7, 6, 5]]
    verts = (verts * (obj_size * .5)) @ cam_R_m2c.T + cam_t_m2c  # [8+1, 3]
    votes = torch.zeros(12, dtype=torch.uint8)

    def inc_vote(i, j):
        if [i, j] in edges:
            votes[edges.index([i, j])] += 1
        else:
            votes[edges.index([j, i])] += 1

    for i0, i1, i2, i3 in faces:
        direction = torch.cross(verts[i1] - verts[i0], verts[i2] - verts[i1])
        if torch.dot(direction, verts[i0]) < 0.:
            inc_vote(i0, i1)
            inc_vote(i1, i2)
            inc_vote(i2, i3)
            inc_vote(i3, i0)
    verts_proj = verts @ cam_K.T  # [8+1, 3]
    verts_proj = verts_proj[:, :-1] / verts_proj[:, -1:]  # [8+1, 2]
    if bbox is not None:
        bbox_size = float(bbox[2:].max() * bbox_zoom_out)
        verts_proj -= bbox[:2] - bbox_size * .5
    verts_proj = verts_proj.detach().cpu().numpy()
    for (i0, i1), vote in zip(edges, votes):
        v0, v1 = verts_proj[i0], verts_proj[i1]
        c0 = [int(vert_colors[i0][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        c1 = [int(vert_colors[i1][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        ec = '#' + ''.join([hex((512 + c0[k] + c1[k]) // 2)[-2:] for k in range(3)])
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], ls='--' if vote >= 2 else '-', c=ec, alpha=alpha)
    ax.scatter(verts_proj[:-1, 0], verts_proj[:-1, 1], c=vert_colors[:-1], zorder=10, alpha=1.)
    ax.scatter(verts_proj[-1, 0], verts_proj[-1, 1], c=vert_colors[-1], marker='+', zorder=10, alpha=1.)
    if bbox is not None and adjust_axis:
        ax.set_xlim(0, bbox_size)
        ax.set_ylim(bbox_size, 0)
        ax.set_aspect('equal')
    return ax
