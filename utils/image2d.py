from typing import Union

import torch
from torch.nn import functional as F

from utils.const import dtype
from utils.io import parse_device


def get_coord_2d_map(width: int, height: int, cam_K: torch.Tensor = None, device: Union[torch.device, str]=None,
                     dtype: torch.dtype = dtype) -> torch.Tensor:
    """
    :param width: int
    :param height: int
    :param cam_K: [3, 3]
    :return: [2(XY), H, W]
    """
    device = cam_K.device if cam_K is not None else parse_device(device)
    dtype = cam_K.dtype if cam_K is not None else dtype
    coord_2d_x, coord_2d_y = torch.meshgrid(
        torch.arange(float(width), dtype=dtype), torch.arange(float(height), dtype=dtype), indexing='xy')  # [H, W]
    coord_2d = torch.stack([coord_2d_x, coord_2d_y, torch.ones_like(coord_2d_x)], dim=0).to(device)  # [3(XY1), H, W]
    if cam_K is not None:
        cam_K /= float(cam_K[-1, -1])
        coord_2d = torch.linalg.solve(cam_K, coord_2d.reshape(3, -1)).reshape(3, height, width)
        # solve(K, M) == K.inv() @ M
    return coord_2d[:2]  # [2(XY), H, W]


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

    :param bbox: [N, 4(XYWH)]
    :param dzi_ratio: [N, 4(XYWH)]
    :return: [N, 4(XYWH)]
    """
    bbox = bbox.clone()
    bbox[:, :2] += bbox[:, 2:] * dzi_ratio[:, :2]  # [N, 2]
    bbox[:, 2:] *= 1. + dzi_ratio[:, 2:]  # [N, 2]
    return bbox


def get_dzi_crop_size(bbox: torch.Tensor, dzi_bbox_zoom_out: Union[torch.Tensor, float] = 1.) -> torch.Tensor:
    """
    dynamic zoom in

    :param bbox: [N, 4(XYWH)]
    :param dzi_bbox_zoom_out: [N] or float
    :return: [N]
    """
    crop_size, _ = bbox[:, 2:].max(dim=-1)
    crop_size *= dzi_bbox_zoom_out
    return crop_size
