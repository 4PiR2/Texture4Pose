import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pytorch3d.transforms
import torch
import torch.nn.functional as F


def normalize_cam_K(cam_K: torch.Tensor) -> torch.Tensor:
    return cam_K / cam_K[..., -1:, -1:]


def t_to_t_site(t: torch.Tensor, bbox: torch.Tensor, cam_K: torch.Tensor, obj_diameter: torch.Tensor) -> torch.Tensor:
    """
    :param t: [..., 3]
    :param bbox: [..., 4(XYWH)]
    :param cam_K: [..., 3, 3]
    :param obj_diameter: [...]
    :return: [..., 3]
    """
    K_bbox = torch.zeros(*bbox.shape[:-1], 3, 3, dtype=bbox.dtype, device=bbox.device)
    K_bbox[..., 0, 0] = bbox[..., 2]
    K_bbox[..., 1, 1] = bbox[..., 3]
    K_bbox[..., -1, -1] = 1.
    K_bbox[..., :-1, -1] = bbox[..., :2]
    f = (cam_K[..., 0, 0] * cam_K[..., 1, 1]) ** .5  # focal_length * cam_K[-1, -1]
    s = (bbox[..., 2] * bbox[..., 3]) ** .5
    fd_s = f * obj_diameter / s
    delta_t = torch.linalg.solve(K_bbox, cam_K @ t[..., None])[..., 0]  # (t_site_x, t_site_y, 1.) * t_z * cam_K[-1, -1]
    t_site_xy = delta_t[..., :-1] / delta_t[..., -1:]
    t_site_z = fd_s / delta_t[..., -1]
    return torch.cat([t_site_xy, t_site_z[..., None]], dim=-1)


def t_site_to_t(t_site: torch.Tensor, bbox: torch.Tensor, cam_K: torch.Tensor, obj_diameter: torch.Tensor) \
        -> torch.Tensor:
    """
    :param t_site: [..., 3]
    :param bbox: [..., 4(XYWH)]
    :param cam_K: [..., 3, 3]
    :param obj_diameter: [...]
    :return: [..., 3]
    """
    K_bbox = torch.zeros(*bbox.shape[:-1], 3, 3, dtype=bbox.dtype, device=bbox.device)
    K_bbox[..., 0, 0] = bbox[..., 2]
    K_bbox[..., 1, 1] = bbox[..., 3]
    K_bbox[..., -1, -1] = 1.
    K_bbox[..., :-1, -1] = bbox[..., :2]
    f = (cam_K[..., 0, 0] * cam_K[..., 1, 1]) ** .5  # focal_length * cam_K[-1, -1]
    s = (bbox[..., 2] * bbox[..., 3]) ** .5
    fd_s = f * obj_diameter / s
    delta_t_z = fd_s / t_site[..., -1]  # t_z * cam_K[-1, -1]
    delta_t = torch.cat([t_site[..., :-1], torch.ones_like(t_site[..., -1:])], dim=-1) * delta_t_z[..., None]
    # (t_site_x, t_site_y, 1.) * t_z * cam_K[-1, -1]
    return torch.linalg.solve(cam_K, K_bbox @ delta_t[..., None])[..., 0]


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
        mask_inlier = torch.zeros_like(mask)
    else:
        mask_inlier = mask
    for i in range(N):
        if mask[i, 0].sum() < 4:
            continue
        x = coord_3d[i].permute(1, 2, 0)[mask[i, 0]].detach().cpu().numpy()
        y = coord_2d[i].permute(1, 2, 0)[mask[i, 0]].detach().cpu().numpy()
        if ransac:
            ret, pred_R_exp, pred_t, inliers = cv2.solvePnPRansac(x, y, np.eye(3), None, iterationsCount=10000,
                                                                reprojectionError=1e-3, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ret:
                continue
            inliers = inliers.flatten()
            m = mask[i, 0].detach().cpu().numpy()
            m_i, m_j = np.where(m)
            m_inlier = np.zeros_like(m)
            m_inlier[m_i[inliers], m_j[inliers]] = 2
            mask_inlier[i, 0] = torch.tensor(m_inlier)
        else:
            ret, pred_R_exp, pred_t = cv2.solvePnP(x, y, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ret:
                continue
        pred_R, _ = cv2.Rodrigues(pred_R_exp)
        pnp_cam_R_m2c[i] = torch.tensor(pred_R)
        pnp_cam_t_m2c[i] = torch.tensor(pred_t.flatten())
    return pnp_cam_R_m2c, pnp_cam_t_m2c, mask_inlier


def show_pose(ax: Axes, cam_K: torch.Tensor, cam_R_m2c: torch.Tensor, cam_t_m2c: torch.Tensor, obj_size: torch.Tensor,
              bbox: torch.Tensor = None, adjust_axis: bool = True, alpha: float = 1.) \
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
        bbox_size = float(bbox[2:].max())
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


def show_pose_mesh_105(ax: Axes, cam_K: torch.Tensor, cam_R_m2c: torch.Tensor, cam_t_m2c: torch.Tensor, obj_size: torch.Tensor = None,
              bbox: torch.Tensor = None, adjust_axis: bool = True, alpha: float = 1.) \
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
    cull_backfaces = 1
    from renderer.cube_mesh import cube, sphericon, cylinder_strip
    # mesh = sphericon(2, 0, device=device)
    mesh = cylinder_strip(2, device=device)
    # mesh = pytorch3d.utils.ico_sphere(level=1).to(device)
    mesh.scale_verts_(.05)
    verts = mesh.verts_packed()  # [N, 3]
    edges = mesh.edges_packed()
    vert_colors = [f'#{hex(r)[-2:]}{hex(g)[-2:]}{hex(b)[-2:]}' for r, g, b in ((verts / .1 + .5) * 255.).round().int() + 512]
    faces = mesh.faces_packed()
    verts = verts @ cam_R_m2c.T + cam_t_m2c  # [N, 3]
    votes_e = torch.zeros(len(edges), dtype=torch.uint8)
    votes_v = torch.zeros(len(verts), dtype=torch.uint8)

    def inc_vote(i0, i1):
        for idx in range(len(edges)):
            if (edges[idx, 0] == i0 and edges[idx, 1] == i1) or (edges[idx, 1] == i0 and edges[idx, 0] == i1):
                votes_e[idx] += 1
                break

    for i0, i1, i2 in faces:
        direction = torch.cross(verts[i1] - verts[i0], verts[i2] - verts[i1])
        if torch.dot(direction, verts[i0]) < 0.:
            inc_vote(i0, i1)
            inc_vote(i1, i2)
            inc_vote(i2, i0)
    verts_proj = verts @ cam_K.T  # [N, 3]
    verts_proj = verts_proj[:, :-1] / verts_proj[:, -1:]  # [N, 2]
    if bbox is not None:
        bbox_size = float(bbox[2:].max())
        verts_proj -= bbox[:2] - bbox_size * .5
    verts_proj = verts_proj.detach().cpu().numpy()
    for (i0, i1), vote_e in zip(edges, votes_e):
        v0, v1 = verts_proj[i0], verts_proj[i1]
        c0 = [int(vert_colors[i0][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        c1 = [int(vert_colors[i1][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        ec = '#' + ''.join([hex((512 + c0[k] + c1[k]) // 2)[-2:] for k in range(3)])
        if vote_e > 0 or not cull_backfaces:
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], ls='-', c=ec, alpha=alpha, lw=4)
            votes_v[i0] += 1
            votes_v[i1] += 1
    #     else:
    #         ax.plot([v0[0], v1[0]], [v0[1], v1[1]], ls='--', c=ec, alpha=0.)
    # verts_mask = votes_v > 0
    # ax.scatter(verts_proj[verts_mask, 0], verts_proj[verts_mask, 1], c='k', zorder=10, alpha=1.)
    # ax.scatter(verts_proj[-1, 0], verts_proj[-1, 1], c=vert_colors[-1], marker='+', zorder=10, alpha=1.)
    if bbox is not None and adjust_axis:
        ax.set_xlim(0, bbox_size)
        ax.set_ylim(bbox_size, 0)
        ax.set_aspect('equal')
    return ax


def show_pose_mesh_104(ax: Axes, cam_K: torch.Tensor, cam_R_m2c: torch.Tensor, cam_t_m2c: torch.Tensor, obj_size: torch.Tensor = None,
              bbox: torch.Tensor = None, adjust_axis: bool = True, alpha: float = 1.) \
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
    from renderer.cube_mesh import cylinder_strip
    mesh = cylinder_strip(2, device=device)
    mesh.scale_verts_(.05)
    verts = mesh.verts_packed()  # [N, 3]
    edges = mesh.edges_packed()
    vert_colors = [f'#{hex(r)[-2:]}{hex(g)[-2:]}{hex(b)[-2:]}' for r, g, b in ((verts / .1 + .5) * 255. * 0.).round().int() + 512]
    faces = mesh.faces_packed()
    verts = verts @ cam_R_m2c.T + cam_t_m2c  # [N, 3]
    votes_e = torch.zeros(len(edges), dtype=torch.uint8)
    votes_v = torch.zeros(len(verts), dtype=torch.uint8)

    def inc_vote(i0, i1):
        for idx in range(len(edges)):
            if (edges[idx, 0] == i0 and edges[idx, 1] == i1) or (edges[idx, 1] == i0 and edges[idx, 0] == i1):
                votes_e[idx] += 1
                break

    verts_proj = verts @ cam_K.T  # [N, 3]
    verts_proj = verts_proj[:, :-1] / verts_proj[:, -1:]  # [N, 2]
    if bbox is not None:
        bbox_size = float(bbox[2:].max())
        verts_proj -= bbox[:2] - bbox_size * .5
    verts_proj = verts_proj.detach().cpu().numpy()

    for i0, i1, i2 in faces:
        direction = torch.cross(verts[i1] - verts[i0], verts[i2] - verts[i1])
        if torch.dot(direction, verts[i0]) < 0.:
            inc_vote(i0, i1)
            inc_vote(i1, i2)
            inc_vote(i2, i0)
            v0, v1, v2 = verts_proj[i0], verts_proj[i1], verts_proj[i2]
            tri = plt.Polygon(np.stack([v0, v1, v2]), color='w', zorder=5)
            ax.add_patch(tri)

    for (i0, i1), vote_e in zip(edges, votes_e):
        v0, v1 = verts_proj[i0], verts_proj[i1]
        c0 = [int(vert_colors[i0][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        c1 = [int(vert_colors[i1][1:][k * 2:k * 2 + 2], 16) for k in range(3)]
        ec = '#' + ''.join([hex((512 + c0[k] + c1[k]) // 2)[-2:] for k in range(3)])
        if vote_e > 0:
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], ls='-', c=ec, alpha=alpha, lw=4, zorder=10)
            votes_v[i0] += 1
            votes_v[i1] += 1
        else:
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], ls='-', c=ec, alpha=alpha, lw=4, zorder=1)

    # verts_mask = votes_v > 0
    # ax.scatter(verts_proj[verts_mask, 0], verts_proj[verts_mask, 1], c='k', zorder=10, alpha=1.)
    # ax.scatter(verts_proj[-1, 0], verts_proj[-1, 1], c=vert_colors[-1], marker='+', zorder=10, alpha=1.)
    if bbox is not None and adjust_axis:
        ax.set_xlim(0, bbox_size)
        ax.set_ylim(bbox_size, 0)
        ax.set_aspect('equal')
    return ax
