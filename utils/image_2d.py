from typing import Union

import torch
from matplotlib import pyplot as plt, patches
from torch.nn import functional as F

from utils.const import dtype, plot_colors
from utils.io import parse_device
from utils.transform_3d import normalize_cam_K


def get_coord_2d_map(width: int, height: int, cam_K: torch.Tensor = None, device: Union[torch.device, str]=None,
                     dtype: torch.dtype = dtype) -> torch.Tensor:
    """
    :param width: int
    :param height: int
    :param cam_K: [..., 3, 3]
    :return: [2(XY), H, W]
    """
    device = cam_K.device if cam_K is not None else parse_device(device)
    dtype = cam_K.dtype if cam_K is not None else dtype
    coord_2d_x, coord_2d_y = torch.meshgrid(
        torch.arange(float(width), dtype=dtype), torch.arange(float(height), dtype=dtype), indexing='xy')  # [H, W]
    coord_2d = torch.stack([coord_2d_x, coord_2d_y, torch.ones_like(coord_2d_x)], dim=0).to(device)  # [3(XY1), H, W]
    if cam_K is not None:
        cam_K = normalize_cam_K(cam_K)
        coord_2d = torch.linalg.solve(cam_K, coord_2d.reshape(3, -1)).reshape(*cam_K.shape[:-2], 3, height, width)
        # solve(K, M) == K.inv() @ M
    return coord_2d[..., :2, :, :]  # [..., 2(XY), H, W]


def get_bbox2d_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    :param mask: [..., H, W]
    :return: [..., 4(XYWH)]
    """
    w_mask = mask.any(dim=-2)  # [..., W]
    h_mask = mask.any(dim=-1)  # [..., H]
    x0 = w_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
    y0 = h_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
    w = w_mask.sum(dim=-1)  # [...]
    h = h_mask.sum(dim=-1)  # [...]
    return torch.stack([x0 + w * .5, y0 + h * .5, w, h], dim=-1)  # [..., 4(XYWH)]


def crop_roi(img: Union[list[torch.Tensor], torch.Tensor], bbox: torch.Tensor,
             in_size: Union[torch.Tensor, float] = None, out_size: Union[list[int], int] = None) \
        -> Union[list[torch.Tensor], torch.Tensor]:
    """

    :param img: list of ([N, C, H, W] or [C, H, W])
    :param bbox: [N, 4(XYWH)]
    :param in_size: [N, 2] or [N, 1] or [N] or float
    :param out_size: [int, int] or [int] or int
    :return: list of [N, C, H, W]
    """
    N = len(bbox)
    device = bbox.device
    dtype = bbox.dtype

    if in_size is None:
        in_size = bbox[:, 2:]
    elif isinstance(in_size, float):
        in_size = torch.full([N, 1], in_size, dtype=dtype, device=device)
    else:
        in_size = in_size.reshape(N, -1)

    if isinstance(out_size, int):
        out_size = [out_size]

    if isinstance(img, torch.Tensor):
        img = [img]
        flag = True
    else:
        flag = False

    C = [i.shape[-3] for i in img]
    cat_img = torch.cat([i.expand(N, -1, -1, -1) for i in img], dim=1)
    H, W = cat_img.shape[-2:]
    wh = torch.tensor([W, H], dtype=dtype, device=device)

    theta = torch.zeros(N, 2, 3, dtype=dtype, device=device)
    theta[:, :, -1] = bbox[:, :2] * 2. / wh - 1.
    theta[:, 0, 0], theta[:, 1, 1] = (in_size / wh).T

    grid = F.affine_grid(theta, [N, sum(C), out_size[-1], out_size[0]], align_corners=False)

    crop_img = F.grid_sample(cat_img, grid, align_corners=False).split(C, dim=1)

    if flag:
        crop_img = crop_img[0]

    return crop_img  # [N, C, H, W]


def get_dzi_bbox(bbox: torch.Tensor, dzi_ratio: torch.Tensor) -> torch.Tensor:
    """
    dynamic zoom in

    :param bbox: [..., 4(XYWH)]
    :param dzi_ratio: [..., 4(XYWH)]
    :return: [..., 4(XYWH)]
    """
    bbox = bbox.clone()
    bbox[..., :2] += bbox[..., 2:] * dzi_ratio[..., :2]  # [..., 2]
    bbox[..., 2:] *= 1. + dzi_ratio[..., 2:]  # [..., 2]
    return bbox


def get_dzi_crop_size(bbox: torch.Tensor, dzi_bbox_zoom_out: Union[torch.Tensor, float] = 1.) -> torch.Tensor:
    """
    dynamic zoom in

    :param bbox: [..., 4(XYWH)]
    :param dzi_bbox_zoom_out: [...] or float
    :return: [...]
    """
    crop_size, _ = bbox[..., 2:].max(dim=-1)
    crop_size *= dzi_bbox_zoom_out
    return crop_size


def normalize_channel(x: torch.Tensor) -> torch.Tensor:
    """

    :param x: [..., H, W]
    :return: [..., H, W] \in [0, 1]
    """
    min_val = x.min(-1)[0].min(-1)[0]  # [...]
    max_val = x.max(-1)[0].max(-1)[0]  # [...]
    return (x - min_val[..., None, None]) / (max_val - min_val)[..., None, None]


def lp_loss(x: torch.Tensor, y: torch.Tensor = None, p: int = 1) -> torch.Tensor:
    """

    :param x: [..., C, H, W]
    :param y: [..., C, H, W]
    :param p: int
    :return: [...]
    """
    if y is not None:
        x = x - y
    if p == 1:
        x = x.abs()
    elif p == 2:
        x = x * x
    else:
        x = x.abs() ** p
    return x.mean(dim=[-3, -2, -1])


def draw_ax(ax: plt.Axes, img_1: torch.Tensor, bg_1: torch.Tensor = None, mask: torch.Tensor = None,
            bboxes: torch.Tensor = None) -> plt.Axes:
    """

    :param ax:
    :param img_1: [C, H, W]
    :param bg_1: [C, H, W]
    :param mask: [*, H, W, *]
    :param bboxes: [N, 4(XYWH)]
    :return: ax
    """
    img_255 = img_1.permute(1, 2, 0)[..., :3] * 255
    if img_255.shape[-1] == 2:
        img_255 = torch.cat([img_255, torch.zeros_like(img_255[..., :1])], dim=-1)
    if bg_1 is not None:
        bg_255 = bg_1.permute(1, 2, 0)[..., :3] * 255
        if mask is not None:
            mask = mask.squeeze()[..., None].bool()
            img_255 = img_255 * mask + bg_255 * ~mask
        else:
            img_255 = img_255 * 0.5 + bg_255 * 0.5

    ax.imshow(img_255.detach().cpu().numpy().astype('uint8'))

    if bboxes is not None:
        def add_bbox(ax, x, y, w, h, text=None, color='red'):
            rect = patches.Rectangle((x - w * .5, y - h * .5), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, text, color=color, size=12, ha='center', va='center')

        if bboxes.dim() < 2:
            bboxes = bboxes[None]
        bboxes = bboxes.detach().cpu().numpy()
        for i in range(len(bboxes)):
            add_bbox(ax, *bboxes[i], text=str(i), color=plot_colors[i % len(plot_colors)])
    return ax


def visualize(x: torch.Tensor) -> None:
    """

    :param x: [N, C, H, W] or [C, H, W] or [H, W]
    """
    if x.dim() == 3:
        x = x[None]
    elif x.dim() == 2:
        x = x[None, None]

    for i in x:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        draw_ax(ax, i)
        plt.show()
