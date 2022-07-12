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


@functional_datapipe('bop_scene')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [], [sf.cam_K, sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.gt_bg],
                         required_attributes=['objects', 'scene_mode', 'path'])
        lmo_mode = 'lmo' in self.path
        path = os.path.join(self.path, 'test_all')
        self._scene_camera: list[dict[str, Any]] = []
        self._scene_gt: list[dict[str, Any]] = []
        self._scene_gt_info: list[dict[str, Any]] = []
        self._scene_id: list[int] = []
        self._data_path: list[str] = []
        for dir in os.listdir(path):
            if not dir.startswith('0'):
                continue
            if not lmo_mode and int(dir) not in self.objects:
                continue
            dir_path = os.path.join(path, dir)
            dir_scene_camera = utils.io.read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            self._scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
            dir_scene_gt = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            self._scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
            self._scene_id += [int(key) for key in dir_scene_gt]
            self._data_path += [dir_path] * len(dir_scene_gt)
            dir_scene_gt_info = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
            self._scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]
        self._iterator = iter(range(self.len))

    def main(self):
        item = next(self._iterator)
        scene_gt = torch.utils.data.dataloader.default_collate(self._scene_gt[item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=self.dtype).reshape(-1, 3, 3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=self.dtype) * BOPMesh.scale
        cam_K = torch.tensor(self._scene_camera[item]['cam_K'], device=self.device)
        cam_K = utils.transform_3d.normalize_cam_K(cam_K.reshape(3, 3)).expand(len(obj_id), -1, -1)
        img_path = os.path.join(self._data_path[item], 'rgb/{:0>6d}.png'.format(self._scene_id[item]))
        gt_bg = utils.io.read_img_file(img_path, dtype=self.dtype, device=self.device).expand(len(obj_id), -1, -1, -1)
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, gt_bg

    @property
    def len(self) -> int:
        return len(self._scene_id)
