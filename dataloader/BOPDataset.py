import os.path
from typing import Any, Iterator, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from pytorch3d.transforms import random_rotations
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import T_co

from dataloader.ObjMesh import BOPMesh, RegularMesh
from dataloader.Sample import Sample
from dataloader.Scene import Scene, SceneBatch, SceneBatchOne
from utils.const import debug_mode, pnp_input_size, img_input_size, gdr_mode, dtype
from utils.io import read_json_file, parse_device, read_img_file, read_depth_img_file
from utils.transform import calc_bbox2d_crop, t_to_t_site, get_coord2d_map


class BOPDataset(IterableDataset):
    def __init__(self, obj_list=None, path=None, transform=None, dtype=dtype, device=None, **kwargs):
        self.obj_list: Union[dict[int, str], list[int]] = obj_list
        self.path: str = path
        self.transform: Any = transform
        self.dtype: torch.dtype = dtype
        self.device: torch.device = parse_device(device)
        self.kwargs: dict[str, Any] = kwargs

        self.objects: dict[int, BOPMesh] = {}
        self.objects_eval: dict[int, BOPMesh] = {}

        self._read_model_info()
        self._read_scene_gt_from_BOP()

    def _read_model_info(self):
        self._lmo_mode: bool = self.path is not None and 'lmo' in self.path
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

    def _read_scene_gt_from_BOP(self):
        self._scene_camera: list[dict[str, Any]] = []
        self._scene_gt: list[dict[str, Any]] = []
        self._scene_gt_info: list[dict[str, Any]] = []
        self._scene_id: list[int] = []
        self._data_path: str = os.path.join(self.path, 'test_all')

        for dir in os.listdir(self._data_path):
            if not dir.startswith('0'):
                continue
            if not self._lmo_mode and int(dir) not in self.obj_list:
                continue
            dir_path = os.path.join(self._data_path, dir)
            dir_scene_camera = read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            self._scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
            dir_scene_gt = read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            self._scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
            self._scene_id += [int(key) for key in dir_scene_gt]
            dir_scene_gt_info = read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
            self._scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]

    def __len__(self) -> int:
        return len(self._scene_gt)

    def __iter__(self) -> Iterator[T_co]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item: int = 0) -> Sample:
        cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img_path = self._get_scene_gt_item(item)
        return self._get_sample(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img_path)

    def _get_scene_gt_item(self, item):
        scene_gt = default_collate(self._scene_gt[item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=self.dtype).reshape(-1, 3,
                                                                                                            3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=self.dtype) * BOPMesh.scale

        cam_K = torch.tensor(self._scene_camera[item]['cam_K'], device=self.device).reshape(3, 3)
        cam_K /= float(cam_K[-1, -1])

        img_path = os.path.join(self._data_path, '{:0>6d}/rgb/{:0>6d}.png'.format(
            2 if self._lmo_mode else obj_id[0], self._scene_id[item]))
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img_path

    def _get_sample(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img_path) -> Sample:
        img = read_img_file(img_path, dtype=self.dtype, device=self.device)
        _, _, height, width = img.shape
        coord2d = get_coord2d_map(width, height, cam_K)
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]

        img, gt_coord3d, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj = \
            self._get_features(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord2d, img)

        if self.transform is not None:
            img = self.transform(img)  # [B, 3(RGB), H, W]

        selected = gt_vis_ratio >= .5  # visibility threshold to select object
        bbox = gt_bbox_vis[selected]  # [N, 4(XYWH)]
        crop_size, pad_size, x0, y0 = calc_bbox2d_crop(bbox)

        def crop(img, out_size=pnp_input_size):
            # [N, C, H, W] or [C, H, W]
            padded_img = vF.pad(img, padding=[pad_size])
            c_imgs = [vF.resized_crop((padded_img[i] if img.dim() > 3 else padded_img)[None],
                                      y0[i], x0[i], crop_size[i], crop_size[i], out_size) for i in
                      range(len(bbox))]
            # [1, C, H, W]
            return torch.cat(c_imgs, dim=0)  # [N, C, H, W]

        # F.interpolate doesn't support bool
        result = Sample(obj_id=obj_id[selected], obj_size=obj_size[selected], cam_K=cam_K,
                        gt_cam_R_m2c=gt_cam_R_m2c[selected], gt_cam_t_m2c=gt_cam_t_m2c[selected],
                        gt_cam_t_m2c_site=t_to_t_site(gt_cam_t_m2c[selected], bbox, pnp_input_size / crop_size, cam_K),
                        coord2d=crop(coord2d), gt_coord3d=crop(gt_coord3d[selected]),
                        gt_mask_vis=crop(gt_mask_vis[selected].to(dtype=torch.uint8)).bool(),
                        gt_mask_obj=crop(gt_mask_obj[selected].to(dtype=torch.uint8)).bool(),
                        img=crop(img.squeeze(), img_input_size),
                        dbg_img=img if debug_mode else None,
                        bbox=bbox,
                        )
        return result

    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord2d, img):
        scene_id = self._scene_id[item]
        o_id = 2 if self._lmo_mode else obj_id[0]
        depth_path = os.path.join(self._data_path, '{:0>6d}/depth/{:0>6d}.png'.format(o_id, scene_id))
        depth_img = read_depth_img_file(depth_path, dtype=self.dtype, device=self.device)  # [1, 1, H, W]
        H, W = depth_img.shape[-2:]
        depth_mask = depth_img != 0.
        depth_img *= self._scene_camera[item]['depth_scale'] * BOPMesh.scale
        depth_img = torch.cat([coord2d * depth_img, depth_img], dim=1)  # [1, 3(XYZ), H, W]
        gt_coord3d_vis = (gt_cam_R_m2c.transpose(-2, -1) @ (depth_img.reshape(1, 3, -1) - gt_cam_t_m2c[..., None])) \
                             .reshape(-1, 3, H, W) * depth_mask

        gt_mask_obj = torch.empty(len(obj_id), 1, H, W).to(self.device, dtype=torch.bool)
        gt_mask_vis = torch.empty_like(gt_mask_obj)
        for i in range(len(obj_id)):
            mask_obj_path = os.path.join(self._data_path, '{:0>6d}/mask/{:0>6d}_{:0>6d}.png'.format(o_id, scene_id, i))
            gt_mask_obj[i] = read_depth_img_file(mask_obj_path, dtype=self.dtype, device=self.device)
            mask_vis_path = os.path.join(self._data_path,
                                         '{:0>6d}/mask_visib/{:0>6d}_{:0>6d}.png'.format(o_id, scene_id, i))
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
        return img, gt_coord3d_vis, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj


class RenderedBOPDataset(BOPDataset):
    def __init__(self, scene_mode, **kwargs):
        self.scene_mode: bool = scene_mode
        super().__init__(**kwargs)

    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord2d, img):
        scene_cls = Scene if self.scene_mode else SceneBatch
        height, width = coord2d.shape[-2:]
        scene = scene_cls(objects=self.objects, cam_K=cam_K, obj_id=obj_id, gt_cam_R_m2c=gt_cam_R_m2c,
                          gt_cam_t_m2c=gt_cam_t_m2c, width=width, height=height)

        a = torch.rand(1) * .5 + .5
        d = torch.rand(1) * (1. - a)
        s = 1. - (a + d)
        ambient = a.expand(3)[None]
        diffuse = d.expand(3)[None]
        specular = s.expand(3)[None]
        direction = torch.randn(3)[None]
        shininess = torch.randint(low=40, high=80, size=(1,))  # shininess: 0-1000
        images = scene.render_scene(ambient=ambient, diffuse=diffuse, specular=specular, direction=direction,
                                    shininess=shininess)
        # [B, 3(RGB), H, W]
        img = images * scene.gt_mask + img * ~scene.gt_mask

        return img, scene.gt_coord3d_obj, scene.gt_vis_ratio, \
               scene.gt_mask_vis, scene.gt_mask_obj, scene.gt_bbox_vis, scene.gt_bbox_obj


class RandomPoseBOPDataset(RenderedBOPDataset):
    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self.__getitem__()

    def _read_scene_gt_from_BOP(self):
        pass

    def _get_scene_gt_item(self, item):
        cam_K = torch.tensor([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]],
                             device=self.device)
        img_path = '/data/lmo/test_all/{:0>6d}/rgb/{:0>6d}.png'.format(2, 3)
        obj_id, gt_cam_R_m2c, gt_cam_t_m2c = self._get_random_poses(cam_K)
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img_path

    def _get_random_poses(self, cam_K, num_obj=None):
        obj_id = torch.tensor([oid for oid in self.obj_list], dtype=torch.uint8, device=self.device)
        if num_obj is not None:
            selected = torch.multinomial(torch.ones_like(obj_id), num_obj, replacement=False).sort()
            obj_id = obj_id[selected]
        else:
            num_obj = len(obj_id)

        radii = torch.tensor([self.objects[int(oid)].radius for oid in obj_id], dtype=self.dtype, device=self.device)
        centers = torch.stack([self.objects[int(oid)].center for oid in obj_id], dim=0)

        box2d_min = torch.linalg.inv(cam_K)[:, -1:].T * 1.  # [1, 3(XY1)], inv(K) @ [0., 0., 1.].T
        box2d_max = torch.linalg.solve(cam_K, torch.tensor([[500.], [480.], [1.]], dtype=self.dtype,
                                                         device=self.device)).T  # [1, 3(XY1)], inv(K) @ [W, H, 1.].T
        depth_min, depth_max = .5, 1.2
        box3d_size = (box2d_max - box2d_min) * depth_min - radii[:, None] * 2.
        box3d_size[:, -1] += depth_max - depth_min
        box3d_min = box2d_min * depth_min - centers + radii[:, None]

        if self.scene_mode:
            triu_indices = torch.triu_indices(num_obj, num_obj, 1)
            mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

        while True:
            gt_cam_t_m2c = torch.rand((num_obj, 3), dtype=self.dtype, device=self.device) * box3d_size + box3d_min
            if not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all():
                break
        gt_cam_R_m2c = random_rotations(num_obj, dtype=self.dtype, device=self.device)  # [N, 3, 3]
        return obj_id, gt_cam_R_m2c, gt_cam_t_m2c


class RandomPoseRegularDataset(RandomPoseBOPDataset):
    def _read_model_info(self):
        BOPMesh.scale = .1
        for obj_id in self.obj_list:
            self.objects[obj_id] = \
                RegularMesh(dtype=dtype, device=self.device, obj_id=int(obj_id), name=self.obj_list[obj_id])
            self.objects_eval[obj_id] = \
                RegularMesh(dtype=dtype, device=self.device, obj_id=int(obj_id), name=self.obj_list[obj_id])
