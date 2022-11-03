import cv2
# import pyzbar.pyzbar
import torch
import torch.nn.functional as F


def decode_barcode(img: torch.Tensor, use_zbar: bool = True):
    if not use_zbar:
        bardet = cv2.barcode_BarcodeDetector()
    flag = False
    if img.ndim == 2:
        img = img[None, None]
        flag = True
    elif img.ndim == 3:
        img = img[None]
        flag = True
    img = (img * 255.).permute(0, 2, 3, 1).detach().cpu().numpy().astype('uint8')
    decoded = []
    for im in img:
        if use_zbar:
            pass
            # ret = pyzbar.pyzbar.decode(im)
            # decoded.append(ret[0].data.decode('UTF-8') if len(ret) else None)
        else:
            ret, decoded_info, decoded_type, corners = bardet.detectAndDecode(im)
            decoded.append(decoded_info[0] if ret else None)
    return decoded[0] if flag else decoded


def _perspective_warp(points: torch.Tensor, img_orig: torch.Tensor, w: int, h: int) -> torch.Tensor:
    dtype = points.dtype
    device = points.device
    map0 = torch.tensor([[[-1., -1.], [1., -1.]],
                         [[-1., 1.], [1., 1.]]], dtype=dtype, device=device).reshape(-1, 2)
    points_2d = points.reshape(-1, 2).to(dtype=dtype)
    P = _getPerspectiveTransform(map0, points_2d)
    map1 = torch.stack(torch.meshgrid(torch.linspace(-1., 1., w, dtype=dtype, device=device),
                                      torch.linspace(-1., 1., h, dtype=dtype, device=device),
                                      indexing='xy'), dim=0).reshape(2, -1)
    map2 = P @ torch.cat([map1, torch.ones_like(map1[:1])], dim=0)
    return (map2[:2] / map2[-1]).reshape(2, h, w)[None].permute(0, 2, 3, 1) * 2. / \
           torch.tensor(img_orig.shape[:-3:-1], device=device) - 1.


def _getPerspectiveTransform(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    assert src.shape == (4, 2) and dst.shape == (4, 2)
    src = torch.cat([src, torch.ones_like(src[:, :1])], dim=-1)
    constraint_mat = torch.zeros(8, 9, dtype=src.dtype, device=src.device)
    for i in range(4):
        constraint_mat[2 * i, 3:6] = - src[i]
        constraint_mat[2 * i, 6:] = dst[i, 1] * src[i]
        constraint_mat[2 * i + 1, :3] = src[i]
        constraint_mat[2 * i + 1, 6:] = - dst[i, 0] * src[i]
    p = torch.linalg.solve(constraint_mat[:, :-1], -constraint_mat[:, -1])
    return torch.cat([p, torch.ones_like(p[:1])]).reshape(3, 3)


def correct_image(points: torch.Tensor, img_orig: torch.Tensor, w: int, h: int) -> torch.Tensor:
    warp = _perspective_warp(points, img_orig, w, h)
    return F.grid_sample(img_orig, warp, align_corners=False)
