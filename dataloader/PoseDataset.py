import os.path
from typing import Any, Iterator, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from pytorch3d.transforms import random_rotations
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from dataloader.ObjMesh import ObjMesh, BOPMesh, RegularMesh
from dataloader.Sample import Sample
from dataloader.Scene import Scene, SceneBatch, SceneBatchOne
from utils.const import debug_mode, pnp_input_size, img_input_size, dtype, img_render_size, \
    vis_ratio_threshold, t_depth_min, t_depth_max, lm_cam_K, dzi_bbox_zoom_out, max_dzi_ratio
from utils.io import read_json_file, parse_device, read_img_file, read_depth_img_file
from utils.transform3d import t_to_t_site
from utils.image2d import get_coord_2d_map, crop_roi, get_dzi_bbox, get_dzi_crop_size


class RandomPoseRegularObjDataset(IterableDataset):
    def __init__(self, obj_list=None, transform=None, dtype=dtype, device=None, scene_mode=True, bg_img_path=None,
                 **kwargs):
        self.obj_list: Union[dict[int, str], list[int]] = obj_list
        self.transform: Any = transform
        self.dtype: torch.dtype = dtype
        self.device: torch.device = parse_device(device)
        self.scene_mode: bool = scene_mode
        self.bg_img_path: str = bg_img_path

        self.objects: dict[int, ObjMesh] = {}
        self.objects_eval: dict[int, ObjMesh] = {}

        self._set_model_meshes()

    def __iter__(self) -> Iterator:
        item = None
        while True:
            cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c = self._get_pose_gt(item)
            img = self._get_bg_img(item)
            yield self._get_sample(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img)

    def _set_model_meshes(self):
        for obj_id in self.obj_list:
            self.objects[obj_id] = \
                RegularMesh(dtype=dtype, device=self.device, obj_id=int(obj_id), name=self.obj_list[obj_id])
        self.objects_eval = self.objects

    def _get_pose_gt(self, item):
        obj_id = torch.tensor([oid for oid in self.obj_list], dtype=torch.uint8, device=self.device)
        num_obj = len(obj_id)
        selected, _ = torch.multinomial(torch.ones(len(obj_id)), num_obj, replacement=False).sort()
        obj_id = obj_id[selected]

        radii = torch.tensor([self.objects[int(oid)].radius for oid in obj_id], dtype=self.dtype, device=self.device)
        centers = torch.stack([self.objects[int(oid)].center for oid in obj_id], dim=0)

        cam_K = lm_cam_K.to(self.device)
        box2d_min = torch.linalg.inv(cam_K)[:, -1:].T  # [1, 3(XY1)], inv(K) @ [0., 0., 1.].T
        box2d_max = torch.linalg.solve(cam_K, torch.tensor([[img_render_size], [img_render_size], [1.]],
                                                           dtype=self.dtype,
                                                           device=self.device)).T  # [1, 3(XY1)], inv(K) @ [W, H, 1.].T
        box3d_size = (box2d_max - box2d_min) * t_depth_min - radii[:, None] * 2.
        box3d_size[:, -1] += t_depth_max - t_depth_min
        box3d_min = box2d_min * t_depth_min - centers + radii[:, None]

        if self.scene_mode:
            triu_indices = torch.triu_indices(num_obj, num_obj, 1)
            mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

        while True:
            gt_cam_t_m2c = torch.rand((num_obj, 3), dtype=self.dtype, device=self.device) * box3d_size + box3d_min
            if not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all():
                break
        gt_cam_R_m2c = random_rotations(num_obj, dtype=self.dtype, device=self.device)  # [N, 3, 3]
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c

    def _get_bg_img(self, item):
        if self.bg_img_path is not None:
            file_names = os.listdir(self.bg_img_path)
            img_path = os.path.join(self.bg_img_path, file_names[int(torch.randint(len(file_names), [1]))])
            img = read_img_file(img_path, dtype=self.dtype, device=self.device)
            img = vF.resize(img, [img_render_size, img_render_size])
        else:
            img = torch.zeros(1, 3, img_render_size, img_render_size, dtype=self.dtype, device=self.device)
        return img

    def _get_sample(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img) -> Sample:
        _, _, height, width = img.shape
        coord_2d = get_coord_2d_map(width, height, cam_K)
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]

        img, gt_coord_3d, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj = \
            self._get_features(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img)

        if self.transform is not None:
            img = self.transform(img)  # [B, 3(RGB), H, W]

        selected = gt_vis_ratio >= vis_ratio_threshold  # visibility threshold to select object
        dzi_ratio = (torch.rand(selected.sum(), 4, dtype=self.dtype, device=self.device) * 2. - 1.) * max_dzi_ratio
        dzi_ratio[:, 3] = dzi_ratio[:, 2]
        bbox = get_dzi_bbox(gt_bbox_vis[selected], dzi_ratio)  # [N, 4(XYWH)]
        crop_size = get_dzi_crop_size(bbox, dzi_bbox_zoom_out)

        crop = lambda img, out_size=pnp_input_size: crop_roi(img, bbox, crop_size, out_size)

        coord_2d_roi, gt_coord_3d_roi, gt_mask_vis_roi, gt_mask_obj_roi = crop([coord_2d, gt_coord_3d[selected],
            gt_mask_vis[selected].to(dtype=self.dtype), gt_mask_obj[selected].to(dtype=self.dtype)])

        # F.interpolate doesn't support bool
        result = Sample(obj_id=obj_id[selected], obj_size=obj_size[selected], cam_K=cam_K,
                        gt_cam_R_m2c=gt_cam_R_m2c[selected], gt_cam_t_m2c=gt_cam_t_m2c[selected],
                        gt_cam_t_m2c_site=t_to_t_site(gt_cam_t_m2c[selected], bbox, pnp_input_size / crop_size, cam_K),
                        coord_2d_roi=coord_2d_roi, gt_coord_3d_roi=gt_coord_3d_roi,
                        gt_mask_vis_roi=gt_mask_vis_roi.round().bool(), gt_mask_obj_roi=gt_mask_obj_roi.round().bool(),
                        img_roi=crop(img.squeeze(), img_input_size), dbg_img=img if debug_mode else None,
                        bbox=bbox, gt_bbox_vis=gt_bbox_vis[selected], gt_bbox_obj=gt_bbox_obj[selected],
                        )
        return result

    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img):
        scene_cls = Scene if self.scene_mode else SceneBatch
        height, width = coord_2d.shape[-2:]
        scene = scene_cls(objects=self.objects, cam_K=cam_K, obj_id=obj_id, gt_cam_R_m2c=gt_cam_R_m2c,
                          gt_cam_t_m2c=gt_cam_t_m2c, width=width, height=height)
        # TODO a d s
        a = torch.rand(1) * .5 + .5  # \in [.5, 1.]
        d = torch.rand(1) * .3  # \in [0., .3]
        s = torch.rand(1) * .2  # \in [0., .2]
        ambient = a.expand(3)[None]
        diffuse = d.expand(3)[None]
        specular = s.expand(3)[None]
        direction = torch.randn(3)[None]
        shininess = torch.randint(low=40, high=80, size=(1,))  # shininess: 0-1000
        images = scene.render_scene(ambient=ambient, diffuse=diffuse, specular=specular, direction=direction,
                                    shininess=shininess)
        # [B, 3(RGB), H, W]
        img = images * scene.gt_mask + img * ~scene.gt_mask

        return img, scene.gt_coord_3d_obj, scene.gt_vis_ratio, \
               scene.gt_mask_vis, scene.gt_mask_obj, scene.gt_bbox_vis, scene.gt_bbox_obj


class RandomPoseBOPObjDataset(RandomPoseRegularObjDataset):
    def __init__(self, path, **kwargs):
        self.path: str = path
        self._lmo_mode: bool = self.path is not None and 'lmo' in self.path
        super().__init__(**kwargs)

    def _set_model_meshes(self):
        path_models = os.path.join(self.path, 'models')
        path_models_eval = os.path.join(self.path, 'models_eval')
        objects_info = read_json_file(os.path.join(path_models, 'models_info.json'))
        objects_info_eval = read_json_file(os.path.join(path_models_eval, 'models_info.json'))
        for obj_id in self.obj_list:
            self.objects[obj_id] = BOPMesh(dtype=dtype, device=self.device,
                                           obj_id=int(obj_id), name=self.obj_list[obj_id], is_eval=False,
                                           mesh_path=os.path.join(path_models, f'obj_{int(obj_id):0>6d}.ply'),
                                           **objects_info[str(obj_id)])
            self.objects_eval[obj_id] = BOPMesh(dtype=dtype, device=self.device,
                                                obj_id=int(obj_id), name=self.obj_list[obj_id], is_eval=True,
                                                mesh_path=os.path.join(path_models_eval, f'obj_{int(obj_id):0>6d}.ply'),
                                                **objects_info_eval[str(obj_id)])


class RenderedPoseBOPObjDataset(RandomPoseBOPObjDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scene_camera: list[dict[str, Any]] = []
        self._scene_gt: list[dict[str, Any]] = []
        self._scene_gt_info: list[dict[str, Any]] = []
        self._scene_id: list[int] = []
        self._data_path: str = os.path.join(self.path, 'test_all')
        self._data_path: list[str] = []
        self._read_scene_gt_from_BOP()

    def __iter__(self) -> Iterator:
        for item in range(len(self._scene_gt)):
            cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c = self._get_pose_gt(item)
            img = self._get_bg_img(item)
            yield self._get_sample(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img)

    def _read_scene_gt_from_BOP(self):
        path = os.path.join(self.path, 'test_all')
        for dir in os.listdir(path):
            if not dir.startswith('0'):
                continue
            if not self._lmo_mode and int(dir) not in self.obj_list:
                continue
            dir_path = os.path.join(path, dir)
            dir_scene_camera = read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            self._scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
            dir_scene_gt = read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            self._scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
            self._scene_id += [int(key) for key in dir_scene_gt]
            self._data_path += [dir_path] * len(dir_scene_gt)
            dir_scene_gt_info = read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
            self._scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]

    def _get_pose_gt(self, item):
        scene_gt = default_collate(self._scene_gt[item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=self.dtype).reshape(-1, 3, 3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=self.dtype) * BOPMesh.scale
        cam_K = torch.tensor(self._scene_camera[item]['cam_K'], device=self.device).reshape(3, 3)
        cam_K /= float(cam_K[-1, -1])
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c

    def _get_bg_img(self, item):
        img_path = os.path.join(self._data_path[item], 'rgb/{:0>6d}.png'.format(self._scene_id[item]))
        img = read_img_file(img_path, dtype=self.dtype, device=self.device)
        return img


class BOPObjDataset(RenderedPoseBOPObjDataset):
    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img):
        data_path = self._data_path[item]
        scene_id = self._scene_id[item]
        depth_path = os.path.join(data_path, 'depth/{:0>6d}.png'.format(scene_id))
        depth_img = read_depth_img_file(depth_path, dtype=self.dtype, device=self.device)  # [1, 1, H, W]
        H, W = depth_img.shape[-2:]
        depth_mask = depth_img != 0.
        depth_img *= self._scene_camera[item]['depth_scale'] * BOPMesh.scale
        depth_img = torch.cat([coord_2d * depth_img, depth_img], dim=1)  # [1, 3(XYZ), H, W]
        gt_coord_3d_vis = (gt_cam_R_m2c.transpose(-2, -1) @ (depth_img.reshape(1, 3, -1) - gt_cam_t_m2c[..., None])) \
                             .reshape(-1, 3, H, W) * depth_mask

        gt_mask_obj = torch.empty(len(obj_id), 1, H, W).to(self.device, dtype=torch.bool)
        gt_mask_vis = torch.empty_like(gt_mask_obj)
        for i in range(len(obj_id)):
            mask_obj_path = os.path.join(data_path, 'mask/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_obj[i] = read_depth_img_file(mask_obj_path, dtype=self.dtype, device=self.device)
            mask_vis_path = os.path.join(data_path, 'mask_visib/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_vis[i] = read_depth_img_file(mask_vis_path, dtype=self.dtype, device=self.device)

        # gt_mask = gt_mask_obj.any(dim=0)[None]

        def cvt_bbox(tensor_list_4):
            bbox = torch.stack(tensor_list_4, dim=-1).to(self.device, dtype=self.dtype)
            bbox[:, :2] += bbox[:, 2:] * .5
            return bbox

        scene_gt_info = default_collate(self._scene_gt_info[item])
        gt_bbox_vis = cvt_bbox(scene_gt_info['bbox_visib'])
        gt_bbox_obj = cvt_bbox(scene_gt_info['bbox_obj'])
        gt_vis_ratio = scene_gt_info['visib_fract'].to(self.device, dtype=self.dtype)
        return img, gt_coord_3d_vis, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj
