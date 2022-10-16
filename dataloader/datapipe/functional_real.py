import os
from typing import Any

import pytorch3d.transforms
import torch
import torch.nn.functional as F
from torch.utils.data import functional_datapipe
import torchvision.transforms.functional as vF

from dataloader.datapipe.helper import SampleMapperIDP
from dataloader.sample import SampleFields as sf
import realworld.barcode
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
    def __init__(self, src_dp: SampleMapperIDP, path: str, ext: str = 'heic', texture: str = ''):
        super().__init__(src_dp, [], [sf.o_item], required_attributes=['objects_eval'])
        self.path = path
        self.ext: str = ext
        self.scene_gt: list[dict[str, Any]] = []
        for dir in os.listdir(path):
            if dir.startswith('0'):
                oid = int(dir)
                if oid not in self.objects_eval:
                    continue
                for img_path in utils.io.list_img_from_dir(os.path.join(path, dir, texture), self.ext):
                    self.scene_gt.append({'obj_id': oid, 'img_path': img_path})
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
        super().__init__(src_dp, [sf.gt_bg, sf.cam_K_orig, sf.o_item], [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
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


@functional_datapipe('get_code_info')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, assert_success: bool = False):
        super().__init__(src_dp, [sf.gt_bg, sf.cam_K_orig, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.o_item], [sf.code_info],
                         [sf.o_item], required_attributes=['scene_gt'])
        self._assert_success: bool = assert_success

    def main(self, gt_bg: torch.Tensor, cam_K_orig: torch.Tensor, gt_cam_R_m2c: torch.Tensor,
             gt_cam_t_m2c: torch.Tensor, o_item: int):
        assert len(gt_bg) == 1
        points = torch.tensor([[[-.5, -.5], [-.5, 1.]],
                               [[1., -.5], [1., 1.]]],
                              dtype=gt_bg.dtype, device=gt_bg.device).reshape(-1, 2)
        points = torch.cat([points, torch.zeros_like(points[:, :1]), torch.ones_like(points[:, :1])], dim=-1)
        P = cam_K_orig @ torch.cat([gt_cam_R_m2c, gt_cam_t_m2c[..., None]], dim=-1)
        points_proj = points @ P.transpose(-2, -1)
        points_proj = points_proj[..., :-1] / points_proj[..., -1:]
        img_new = realworld.barcode.correct_image(points_proj, gt_bg, w=1000, h=1000)
        code_info = realworld.barcode.decode_barcode(img_new, use_zbar=False)
        if self._assert_success:
            for i in range(len(code_info)):
                if code_info[i] is None or len(code_info[i]) <= 0:
                    print(o_item)
                    print(self.scene_gt[o_item])
                    utils.image_2d.visualize(gt_bg[i])
                    raise RuntimeError
        return code_info


@functional_datapipe('offset_pose_cylinder')
class _(SampleMapperIDP):
    # details: see weekly report meeting0812
    def __init__(self, src_dp: SampleMapperIDP, scale_true: float, align_x: float, align_y: float):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.obj_size, sf.code_info],
                         [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         required_attributes=['dtype', 'device', 'board'])
        self._size_true: float = scale_true * 2.
        square_length = self.board.board.getSquareLength()
        self._dR: torch.Tensor = torch.eye(3, dtype=self.dtype, device=self.device)
        self._dR_x_180: torch.Tensor = pytorch3d.transforms.euler_angles_to_matrix(
            torch.tensor([0., 0., torch.pi], dtype=self.dtype, device=self.device), 'ZYX') @ self._dR
        self._dt: torch.Tensor = torch.tensor(
            [align_x * square_length + scale_true, align_y * square_length, scale_true],
            dtype=self.dtype, device=self.device)  # (dx + r, dy, h * .5)

    def main(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor, obj_size: torch.Tensor,
             code_info: list):
        m = obj_id == 104
        gt_cam_t_m2c[m] = \
            (gt_cam_t_m2c[m] + (gt_cam_R_m2c[m] @ self._dt[..., None])[..., 0]) * (obj_size[m, :1] / self._size_true)
        dR = torch.stack([self._dR_x_180 if info is not None and int(info[6]) > 0 else self._dR for info in code_info])
        gt_cam_R_m2c[m] = gt_cam_R_m2c[m] @ dR
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('offset_pose_sphericon')
class _(SampleMapperIDP):
    # details: see weekly report meeting0930
    def __init__(self, src_dp: SampleMapperIDP, scale_true: float, align_x: float, align_y: float):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.obj_size, sf.code_info],
                         [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         required_attributes=['dtype', 'device', 'board'])
        self._size_true: float = scale_true * 2.
        square_length = self.board.board.getSquareLength()
        self._dR: torch.Tensor = torch.tensor([[-.5 ** .5, .5 ** .5, 0.],
                                               [0., 0., -1.],
                                               [-.5 ** .5, -.5 ** .5, 0.]], dtype=self.dtype, device=self.device)
        self._dR_x_180: torch.Tensor = pytorch3d.transforms.euler_angles_to_matrix(
            torch.tensor([0., 0., torch.pi], dtype=self.dtype, device=self.device), 'ZYX') @ self._dR
        self._dt: torch.Tensor = torch.tensor(
            [align_x * square_length + .5 ** .5 * scale_true, align_y * square_length, .5 ** .5 * scale_true],
            dtype=self.dtype, device=self.device)  # (dx + r * sqrt(1/2), dy, r * sqrt(1/2))

    def main(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor, obj_size: torch.Tensor,
             code_info: list):
        m = obj_id == 105
        gt_cam_t_m2c[m] = \
            (gt_cam_t_m2c[m] + (gt_cam_R_m2c[m] @ self._dt[..., None])[..., 0]) * (obj_size[m, :1] / self._size_true)
        dR = torch.stack([self._dR_x_180 if info is not None and int(info[6]) > 0 else self._dR for info in code_info])
        gt_cam_R_m2c[m] = gt_cam_R_m2c[m] @ dR
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('augment_pose')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, keep_first: int = 0, t_depth_range: tuple[float, float] = (.5, 1.2),
                 t_center_range: tuple[float, float] = (-.7, .7), cuboid: bool = False, depth_max_try: int = 100,
                 batch: int = None):
        super().__init__(src_dp, [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.gt_cam_R_m2c_aug])
        self._keep_first: int = keep_first
        self._t_depth_range: tuple[float, float] = t_depth_range
        self._t_center_range: tuple[float, float] = t_center_range
        self._cuboid: bool = cuboid
        self._depth_max_try: int = int(depth_max_try)
        self._B: int = batch

    def main(self, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor):
        dtype = gt_cam_R_m2c.dtype
        device = gt_cam_R_m2c.device
        N = len(gt_cam_R_m2c)
        B = self._B if self._B else N
        gt_cam_R_m2c_aug = [torch.eye(3, dtype=dtype, device=device).expand(min(self._keep_first, N), -1, -1)]
        for i in range(self._keep_first, N, B):
            max_try_depth = self._depth_max_try
            while True:
                max_try_depth -= 1
                dR = pytorch3d.transforms.random_rotations(B, dtype=dtype, device=device)
                t = (dR @ gt_cam_t_m2c[i:i+B, ..., None])[..., 0]
                if not self._cuboid:
                    t[..., :-1] /= t[..., -1:]
                if max_try_depth >= 0:
                    t_depth_ok = (self._t_depth_range[0] <= t[..., -1]) & (t[..., -1] <= self._t_depth_range[-1])
                else:
                    t_depth_ok = t[..., -1] >= 0.
                t_center_ok = (self._t_center_range[0] <= t[..., :-1]) & (t[..., :-1] <= self._t_center_range[-1])
                if t_depth_ok.all() and t_center_ok.all():
                    break
            gt_cam_t_m2c[i:i+B] = (dR @ gt_cam_t_m2c[i:i+B, ..., None])[..., 0]
            gt_cam_R_m2c[i:i+B] = dR @ gt_cam_R_m2c[i:i+B]
            gt_cam_R_m2c_aug.append(dR)
        return gt_cam_R_m2c, gt_cam_t_m2c, torch.cat(gt_cam_R_m2c_aug, dim=0)


@functional_datapipe('crop_roi_real_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        fields = [sf.gt_bg]
        super().__init__(src_dp, [sf.coord_2d_roi, sf.gt_cam_R_m2c_aug, sf.cam_K_orig] + fields, fields, [sf.gt_cam_R_m2c_aug],
                         required_attributes=['img_render_size'])

    def main(self, coord_2d_roi: torch.Tensor, gt_cam_R_m2c_aug: torch.Tensor, cam_K_orig: torch.Tensor,
             gt_bg: torch.Tensor):
        dtype = gt_bg.dtype
        device = gt_bg.device
        N, _, H, W = gt_bg.shape
        grid_K = torch.tensor([[2. / (W - 1), 0., -1.],
                               [0., 2. / (H - 1), -1.],
                               [0., 0., 1.]], dtype=dtype, device=device)
        coord_2d_homo = torch.cat([coord_2d_roi, torch.ones_like(coord_2d_roi[..., :1, :, :])], dim=-3)
        grid_homo = (grid_K @ cam_K_orig @ gt_cam_R_m2c_aug.transpose(-2, -1) @ coord_2d_homo.reshape(N, 3, -1)) \
            .reshape(N, 3, self.img_render_size, self.img_render_size)
        grid = grid_homo[..., :-1, :, :] / grid_homo[..., -1:, :, :]
        gt_bg = F.grid_sample(gt_bg, grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)
        return gt_bg


@functional_datapipe('bg_as_real_img')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, delete_original: bool = True):
        super().__init__(src_dp, [sf.gt_bg_roi], [sf.img_roi], [sf.gt_bg_roi] if delete_original else [])

    def main(self, gt_bg_roi: torch.Tensor):
        return gt_bg_roi


@functional_datapipe('rand_occlude_apply_real')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.img_roi, sf.gt_mask_vis_roi, sf.gt_mask_obj_roi], [sf.img_roi])

    def main(self, img_roi: torch.Tensor, gt_mask_vis_roi: torch.Tensor, gt_mask_obj_roi: torch.Tensor):
        occlusion = gt_mask_vis_roi ^ gt_mask_obj_roi
        return img_roi * ~occlusion
