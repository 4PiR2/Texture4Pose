from typing import Union

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as vF

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


def crop_roi(img: torch.Tensor, bbox: torch.Tensor,
             in_size: Union[torch.Tensor, int] = None, out_size: Union[torch.Tensor, int] = None) -> torch.Tensor:
    """

    :param img: [N, C, H, W] or [C, H, W]
    :param bbox: [N, 4(XYWH)]
    :param in_size: [N, 2] or [N]
    :param out_size: [N, 2] or [N]
    :return: [N, C, H, W]
    """
    N = len(bbox)
    C, H, W = img.shape[-3:]
    if in_size is None:
        in_size = bbox[:, 2:].int()
    elif isinstance(in_size, int):
        in_size = torch.full([N, 1], in_size, dtype=torch.int)
    else:
        in_size = in_size.reshape(N, -1)
    if out_size is None:
        out_size = bbox[:, 2:].int()
    elif isinstance(out_size, int):
        out_size = torch.full([N, 1], out_size, dtype=torch.int)
    else:
        out_size = out_size.reshape(N, -1)

    x0, y0 = (bbox[:, :2] - in_size * .5).round().int().T
    x1, y1 = (bbox[:, :2] + in_size * .5).round().int().T
    pad_left = F.relu(-x0).max()
    pad_right = F.relu(x1 - W + 1).max()
    pad_top = F.relu(-y0).max()
    pad_bottom = F.relu(y1 - H + 1).max()

    padded_img = vF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom])
    x0 += pad_left
    y0 += pad_top
    c_imgs = [vF.resized_crop((padded_img[i] if img.dim() > 3 else padded_img)[None], y0[i], x0[i],
              in_size[i, -1], in_size[i, 0], [int(out_size[i, -1]), int(out_size[i, 0])]) for i in range(len(bbox))]
    # list of [1, C, H, W]
    return torch.cat(c_imgs, dim=0)  # [N, C, H, W]


def get_dzi_shifted_bbox(bbox, shift_ratio_wh: torch.Tensor = torch.zeros(2)):
    """
    dynamic zoom in

    :param bbox: [N, 4(XYWH)]
    :param shift_ratio_wh: [2(RxRy)]
    :return: [N, 4(XYWH)]
    """
    bbox = bbox.clone()
    bbox[:, :2] += bbox[:, 2:] * shift_ratio_wh.to(bbox.device, dtype=bbox.dtype)  # [N, 2]
    return bbox


def get_dzi_crop_size(bbox, bbox_zoom_out: float = 1.):
    """
    dynamic zoom in

    :param bbox: [N, 4(XYWH)]
    :param bbox_zoom_out: float
    :return: [N]
    """
    crop_size, _ = bbox[:, 2:].max(dim=-1)
    crop_size *= bbox_zoom_out
    return crop_size.round().int()
