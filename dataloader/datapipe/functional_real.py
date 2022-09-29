import os
from typing import Any

import pytorch3d.transforms
import torch
import torch.nn.functional as F
from torch.utils.data import functional_datapipe
import torchvision.transforms.functional as vF

from dataloader.datapipe.helper import SampleMapperIDP
from dataloader.sample import SampleFields as sf
import realworld.charuco_board
import utils.image_2d
import utils.io
import utils.transform_3d


@functional_datapipe('rand_gt_rotation_real')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, cylinder_strip_thresh_theta: float = 15. * torch.pi / 180.):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_t_m2c], [sf.gt_cam_R_m2c], required_attributes=['dtype', 'device'])
        self._cos_thresh = torch.tensor(cylinder_strip_thresh_theta).cos()

    def main(self, obj_id: torch.Tensor, gt_cam_t_m2c: torch.Tensor):
        # for cylinderstrip, empty projection in z-axis direction
        gt_cam_R_m2c = []
        for oid, t in zip(obj_id, F.normalize(gt_cam_t_m2c, p=2, dim=-1)):
            while True:
                rot = pytorch3d.transforms.random_rotations(1, dtype=self.dtype, device=self.device)[0]
                if oid != 104 or torch.dot(t, rot[:, 2]).abs() <= self._cos_thresh:
                    gt_cam_R_m2c.append(rot)
                    break
        return torch.stack(gt_cam_R_m2c, dim=0)


@functional_datapipe('load_real_scene')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, path: str, ext: str = 'heic'):
        super().__init__(src_dp, [], [sf.o_item])
        self.path = path
        self.ext: str = ext
        self.scene_gt: list[dict[str, Any]] = []
        for dir in os.listdir(path):
            if dir.startswith('0'):
                for img_path in utils.io.list_img_from_dir(os.path.join(path, dir), self.ext):
                    self.scene_gt.append({'obj_id': int(dir), 'img_path': img_path})
        self._iterator = iter(range(self.len))

    @property
    def len(self) -> int:
        return len(self.scene_gt)

    def main(self):
        return next(self._iterator)


@functional_datapipe('set_real_scene')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.o_item], [sf.obj_id, sf.gt_bg], required_attributes=['dtype', 'device'])

    def main(self, o_item: int):
        scene = self.scene_gt[o_item]
        obj_id = torch.tensor([scene['obj_id']], dtype=torch.uint8, device=self.device)  # [1]
        img = utils.io.read_img_file(scene['img_path'], dtype=self.dtype, device=self.device)  # [1, 3(RGB), H, W]
        return obj_id, img


@functional_datapipe('set_calib_camera')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, w_square: int, h_square: int, square_length: float,
                 cam_K: torch.Tensor = None):
        super().__init__(src_dp, [sf.N], [sf.cam_K_orig], required_attributes=['dtype', 'device', 'path'])
        self.board: realworld.charuco_board.ChArUcoBoard = \
            realworld.charuco_board.ChArUcoBoard(w_square, h_square, square_length)
        if cam_K is None:
            cam_K = self.board.calibrate_camera(
                utils.io.list_img_from_dir(os.path.join(self.path, 'calib'), ext=self.ext))[0]
        self._cam_K: torch.Tensor = utils.transform_3d.normalize_cam_K(
            torch.tensor(cam_K, dtype=self.dtype, device=self.device))  # [3, 3]

    def main(self, N: int):
        return self._cam_K.expand(N, -1, -1)  # [N, 3, 3]


@functional_datapipe('estimate_pose')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_bg, sf.cam_K_orig, sf.o_item], [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], [sf.o_item],
                         required_attributes=['board', 'scene_gt'])

    def main(self, gt_bg: torch.Tensor, cam_K_orig: torch.Tensor, o_item: int):
        device = gt_bg.device
        dtype = gt_bg.dtype
        img_gray = vF.rgb_to_grayscale(gt_bg)  # [N, 1, H, W]
        gt_cam_R_m2c = []
        gt_cam_t_m2c = []
        for i, k in zip(img_gray, cam_K_orig):
            ret, p_rmat, p_tvec, _ = \
                self.board.estimate_pose((i[0] * 255.).detach().cpu().numpy().astype('uint8'), k.detach().cpu().numpy())
            if not ret:
                print(o_item)
                print(self.scene_gt[o_item])
                utils.image_2d.visualize(img_gray)
                raise RuntimeError
            gt_cam_R_m2c.append(p_rmat)
            gt_cam_t_m2c.append(p_tvec[..., 0])
        gt_cam_R_m2c = torch.tensor(gt_cam_R_m2c, dtype=dtype, device=device)
        gt_cam_t_m2c = torch.tensor(gt_cam_t_m2c, dtype=dtype, device=device)
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('offset_pose_cylinder')
class _(SampleMapperIDP):
    # details: see weekly report meeting0812
    def __init__(self, src_dp: SampleMapperIDP, scale_true: float, align_x: float, align_y: float):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.obj_size],
                         [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], required_attributes=['dtype', 'device', 'objects', 'board'])
        self._size_true: float = scale_true * 2.
        square_length = self.board.board.getSquareLength()
        self._dt: torch.Tensor = torch.tensor(
            [align_x * square_length + scale_true, align_y * square_length, scale_true],
            dtype=self.dtype, device=self.device)  # (dx + r, dy, h * .5)

    def main(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor, obj_size: torch.Tensor):
        m = obj_id == 104
        gt_cam_t_m2c[m] = \
            (gt_cam_t_m2c[m] + (gt_cam_R_m2c[m] @ self._dt[..., None])[..., 0]) * (obj_size[m, :1] / self._size_true)
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('offset_pose_sphericon')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, scale_true: float, align_x: float, align_y: float):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.obj_size],
                         [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], required_attributes=['dtype', 'device', 'objects', 'board'])
        self._size_true: float = scale_true * 2.
        square_length = self.board.board.getSquareLength()
        self._dR: torch.Tensor = torch.tensor([[-.5 ** .5, .5 ** .5, 0.],
                                               [0., 0., -1.],
                                               [-.5 ** .5, -.5 ** .5, 0.]], dtype=self.dtype, device=self.device)
        self._dt: torch.Tensor = torch.tensor(
            [align_x * square_length + .5 ** .5 * scale_true, align_y * square_length, .5 ** .5 * scale_true],
            dtype=self.dtype, device=self.device)  # (dx + r * sqrt(1/2), dy, r * sqrt(1/2))

    def main(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor, obj_size: torch.Tensor):
        m = obj_id == 105
        gt_cam_t_m2c[m] = \
            (gt_cam_t_m2c[m] + (gt_cam_R_m2c[m] @ self._dt[..., None])[..., 0]) * (obj_size[m, :1] / self._size_true)
        gt_cam_R_m2c[m] = gt_cam_R_m2c[m] @ self._dR
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('crop_roi_real_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        fields = [sf.gt_bg]
        super().__init__(src_dp, [sf.bbox, sf.cam_K_orig] + fields, fields, required_attributes=['img_render_size'])

    def main(self, bbox: torch.Tensor, cam_K_orig: torch.Tensor, gt_bg: torch.Tensor):
        fx = cam_K_orig[..., 0, 0]
        fy = cam_K_orig[..., 1, 1]
        px = cam_K_orig[..., 0, -1]
        py = cam_K_orig[..., 1, -1]
        bbox = bbox.clone()
        bbox[..., 0] = bbox[..., 0] * fx + px
        bbox[..., 1] = bbox[..., 1] * fy + py
        bbox[..., 2] *= fx
        bbox[..., 3] *= fy
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self.img_render_size, mode)
        return crop(gt_bg, 'bilinear')


@functional_datapipe('bg_as_real_img')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, delete_original: bool = True):
        super().__init__(src_dp, [sf.gt_bg_roi], [sf.img_roi], [sf.gt_bg_roi] if delete_original else [])

    def main(self, gt_bg_roi: torch.Tensor):
        return gt_bg_roi
