import os
from typing import Union, Any

import torch
from torch.utils.data import functional_datapipe

from dataloader.obj_mesh import ObjMesh, BOPMesh
import dataloader.datapipe.functional_basic
from dataloader.datapipe.helper import SampleMapperIDP, IterDataPipe
from dataloader.sample import SampleFields as sf, Sample
import utils.io
import utils.image_2d
import utils.transform_3d


@functional_datapipe('init_bop_objects')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, obj_list: Union[dict[int, str], list[int]], path: str):
        super().__init__(src_dp, [], [sf.obj_id], required_attributes=['dtype', 'device', 'objects', 'objects_eval'])

        path_models = os.path.join(path, 'models')
        path_models_eval = os.path.join(path, 'models_eval')
        objects_info = utils.io.read_json_file(os.path.join(path_models, 'models_info.json'))
        objects_info_eval = utils.io.read_json_file(os.path.join(path_models_eval, 'models_info.json'))
        objects = {}
        objects_eval = {}
        for obj_id in obj_list:
            objects[obj_id] = BOPMesh(
                dtype=self.dtype, device=self.device, obj_id=int(obj_id), name=obj_list[obj_id], is_eval=False,
                mesh_path=os.path.join(path_models, f'obj_{int(obj_id):0>6d}.ply'),  **objects_info[str(obj_id)])
            objects_eval[obj_id] = BOPMesh(
                dtype=self.dtype, device=self.device, obj_id=int(obj_id), name=obj_list[obj_id], is_eval=True,
                mesh_path=os.path.join(path_models_eval, f'obj_{int(obj_id):0>6d}.ply'),
                **objects_info_eval[str(obj_id)])

        self.path: str = path
        self.objects: dict[int, ObjMesh] = {**self.objects, **objects}
        self.objects_eval: dict[int, ObjMesh] = {**self.objects_eval, **objects}

    def main(self):
        obj_id = torch.tensor(list(self.objects), dtype=torch.uint8, device=self.device)
        return obj_id


def init_objects(src_dp: IterDataPipe[Sample], obj_list: Union[dict[int, str], list[int]], path: str = None):
    if isinstance(obj_list, dict):
        obj_list_bop = {}
        obj_list_regular = {}
        for k, v in obj_list.items():
            if k < 100:
                obj_list_bop[k] = v
            else:
                obj_list_regular[k] = v
    else:
        obj_list_bop = []
        obj_list_regular = []
        for k in obj_list:
            if k < 100:
                obj_list_bop.append(k)
            else:
                obj_list_regular.append(k)
    dp = src_dp.init_regular_objects(obj_list_regular)
    if obj_list_bop:
        assert path is not None
        dp = dp.init_bop_objects(obj_list_bop, path)
    return dp


@functional_datapipe('load_bop_scene')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [], [sf.o_item], required_attributes=['objects', 'path'])
        lmo_mode = 'lmo' in self.path
        path = os.path.join(self.path, 'test_all')
        self.scene_camera: list[dict[str, Any]] = []
        self.scene_gt: list[dict[str, Any]] = []
        self.scene_gt_info: list[dict[str, Any]] = []
        self.scene_id: list[int] = []
        self.data_path: list[str] = []
        for dir in os.listdir(path):
            if not dir.startswith('0'):
                continue
            if not lmo_mode and int(dir) not in self.objects:
                continue
            dir_path = os.path.join(path, dir)
            dir_scene_camera = utils.io.read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            self.scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
            dir_scene_gt = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            self.scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
            self.scene_id += [int(key) for key in dir_scene_gt]
            self.data_path += [dir_path] * len(dir_scene_gt)
            dir_scene_gt_info = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
            self.scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]
        self._iterator = iter(range(self.len))

    @property
    def len(self) -> int:
        return len(self.scene_id)

    def main(self):
        return next(self._iterator)


@functional_datapipe('rand_scene_id')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [], [sf.o_item], required_attributes=['len'])

    def main(self):
        return int(torch.randint(self.len, [1]))


@functional_datapipe('set_pose')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.o_item], [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         required_attributes=['dtype', 'device', 'scene_gt'])

    def main(self, o_item: int):
        scene_gt = torch.utils.data.dataloader.default_collate(self.scene_gt[o_item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=self.dtype).reshape(-1, 3, 3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=self.dtype) * BOPMesh.scale
        return obj_id, gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('set_camera')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N, sf.o_item], [sf.cam_K], required_attributes=['scene_camera'])

    def main(self, N: int, o_item: int):
        cam_K = torch.tensor(self.scene_camera[o_item]['cam_K'], device=self.device)
        cam_K = utils.transform_3d.normalize_cam_K(cam_K.reshape(3, 3)).expand(N, -1, -1)
        return cam_K


@functional_datapipe('set_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N, sf.o_item], [sf.gt_bg],
                         required_attributes=['dtype', 'device', 'data_path', 'scene_id'])

    def main(self, N: int, o_item: int):
        img_path = os.path.join(self.data_path[o_item], 'rgb/{:0>6d}.png'.format(self.scene_id[o_item]))
        gt_bg = utils.io.read_img_file(img_path, dtype=self.dtype, device=self.device).expand(N, -1, -1, -1)
        return gt_bg


@functional_datapipe('remove_item_id')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [], [sf.o_item])


@functional_datapipe('set_depth')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N, sf.o_item], [sf.gt_coord_3d],
                         required_attributes=['dtype', 'device', 'data_path', 'scene_id', 'scene_camera'])

    def main(self, N: int, o_item: int):
        depth_path = os.path.join(self.data_path[o_item], 'depth/{:0>6d}.png'.format(self.scene_id[o_item]))
        gt_depth = utils.io.read_depth_img_file(depth_path, dtype=self.dtype, device=self.device)  # [1, 1, H, W]
        gt_depth *= self.scene_camera[o_item]['depth_scale'] * BOPMesh.scale
        return gt_depth.expand(N, -1, -1, -1)


@functional_datapipe('set_mask')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, bitwise_and_with_existing: bool = False):
        super().__init__(src_dp, [sf.N, sf.o_item, sf.gt_mask_vis, sf.gt_mask_obj] if bitwise_and_with_existing else
                         [sf.N, sf.o_item], [sf.gt_mask_vis, sf.gt_mask_obj],
                         required_attributes=['dtype', 'device', 'data_path', 'scene_id'])

    def main(self, N: int, o_item: int, gt_mask_vis: torch.Tensor = None, gt_mask_obj: torch.Tensor = None):
        data_path = self.data_path[o_item]
        scene_id = self.scene_id[o_item]
        gt_mask_vis_list = []
        gt_mask_obj_list = []
        for i in range(N):
            mask_vis_path = os.path.join(data_path, 'mask_visib/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_vis_list.append(utils.io.read_depth_img_file(mask_vis_path, dtype=self.dtype, device=self.device))
            mask_obj_path = os.path.join(data_path, 'mask/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_obj_list.append(utils.io.read_depth_img_file(mask_obj_path, dtype=self.dtype, device=self.device))
        gt_mask_vis_bop = torch.cat(gt_mask_vis_list, dim=0).bool()
        gt_mask_obj_bop = torch.cat(gt_mask_obj_list, dim=0).bool()
        H, W = gt_mask_vis_bop.shape[-2:]
        if gt_mask_vis is not None:
            gt_mask_vis_bop &= gt_mask_vis[..., :H, :W]
        if gt_mask_obj is not None:
            gt_mask_obj_bop &= gt_mask_obj[..., :H, :W]
        # gt_mask = gt_mask_obj_bop.any(dim=0)[None]
        return gt_mask_vis_bop, gt_mask_obj_bop


@functional_datapipe('set_bbox')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N, sf.o_item], [sf.gt_bbox_vis, sf.gt_vis_ratio],
                         required_attributes=['dtype', 'device', 'scene_gt_info'])

    def main(self, N: int, o_item: int):
        def cvt_bbox(tensor_list_4):
            bbox = torch.stack(tensor_list_4, dim=-1).to(self.device, dtype=self.dtype)
            bbox[:, :2] += bbox[:, 2:] * .5
            return bbox

        scene_gt_info = torch.utils.data.dataloader.default_collate(self.scene_gt_info[o_item])
        gt_bbox_vis = cvt_bbox(scene_gt_info['bbox_visib'])
        # gt_bbox_obj = cvt_bbox(scene_gt_info['bbox_obj'])
        gt_vis_ratio = scene_gt_info['visib_fract'].to(self.device, dtype=self.dtype)
        return gt_bbox_vis, gt_vis_ratio


@functional_datapipe('crop_roi_bop')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, out_size: Union[list[int], int], delete_original: bool = True):
        fields = [sf.gt_mask_vis, sf.gt_mask_obj, sf.gt_coord_3d, sf.coord_2d, sf.gt_bg]
        super().__init__(src_dp, [sf.bbox] + fields, [f'{field}_roi' for field in fields[:-1]] + [sf.img_roi],
                         fields if delete_original else [])
        self._out_size: Union[list[int], int] = out_size

    def main(self, bbox: torch.Tensor, gt_mask_vis: torch.Tensor, gt_mask_obj: torch.Tensor, gt_coord_3d: torch.Tensor,
             coord_2d: torch.Tensor, gt_bg: torch.Tensor):
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self._out_size, mode) \
            if img is not None else None

        gt_mask_vis_roi = crop(gt_mask_vis, 'nearest')
        gt_mask_obj_roi = crop(gt_mask_obj, 'nearest')
        gt_coord_3d_roi = crop(gt_coord_3d, 'bilinear')
        coord_2d_roi = crop(coord_2d, 'bilinear')
        img_roi = crop(gt_bg, 'bilinear')
        return gt_mask_vis_roi, gt_mask_obj_roi, gt_coord_3d_roi, coord_2d_roi, img_roi


@functional_datapipe('set_coord_3d')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_coord_3d_roi, sf.coord_2d_roi, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         [sf.gt_coord_3d_roi])

    def main(self, gt_coord_3d_roi: torch.Tensor, coord_2d_roi: torch.Tensor, gt_cam_R_m2c: torch.Tensor,
             gt_cam_t_m2c: torch.Tensor):
        N, _, H, W = coord_2d_roi.shape
        depth_mask = gt_coord_3d_roi.bool()
        depth_img = torch.cat([coord_2d_roi * gt_coord_3d_roi, gt_coord_3d_roi], dim=1)  # [N, 3(XYZ), H, W]
        gt_coord_3d_roi = (gt_cam_R_m2c.transpose(-2, -1) @ (depth_img.reshape(N, 3, -1) - gt_cam_t_m2c[..., None])) \
                              .reshape(-1, 3, H, W) * depth_mask
        return gt_coord_3d_roi
