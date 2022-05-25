import os.path
from typing import Any, Iterator, Union

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as vF
import pytorch3d.transforms

import config.const as cc
from dataloader.obj_mesh import ObjMesh, BOPMesh, RegularMesh
from dataloader.sample import Sample
from dataloader.scene import Scene, SceneBatch, SceneBatchOne
import utils.io
import utils.image_2d
import utils.transform_3d


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.IterableDataset, len: int):
        self.dataset_iter: Iterator = iter(dataset)
        self.len: int = len

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int = None) -> Sample:
        return self.dataset_iter.__next__()


class RandomPoseRegularObjDataset(torch.utils.data.IterableDataset):
    def __init__(self, obj_list=None, transform=None, dtype=cc.dtype, device=cc.device, scene_mode=True,
                 bg_img_path=None, img_render_size=512, img_input_size=256, pnp_input_size=64, cam_K=cc.lm_cam_K,
                 random_t_depth_range=(.8, 1.2), vis_ratio_filter_threshold=.5, max_dzi_ratio=.25,
                 bbox_zoom_out_ratio=1.5, light_ambient_range=(.5, 1.), light_diffuse_range=(0., .3),
                 light_specular_range=(0., .2), light_shininess_range=(40, 80), light_color_range=(1., 1.), **kwargs):
        self.obj_list: Union[dict[int, str], list[int]] = obj_list
        self.transform: Any = transform
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device
        self.scene_mode: bool = scene_mode
        self._bg_img_path: list[str] = [os.path.join(bg_img_path, name) for name in os.listdir(bg_img_path)] \
            if bg_img_path is not None else None

        self.img_render_size: int = img_render_size
        self.img_input_size: int = img_input_size
        self.pnp_input_size: int = pnp_input_size
        self.cam_K: torch.Tensor = cam_K  # [3, 3]
        self.random_t_depth_range: tuple[float, float] = random_t_depth_range
        self.vis_ratio_filter_threshold: float = vis_ratio_filter_threshold
        self.max_dzi_ratio: float = max_dzi_ratio
        self.bbox_zoom_out_ratio: float = bbox_zoom_out_ratio
        self.light_ambient_range: tuple[float, float] = light_ambient_range  # \in [0., 1.]
        self.light_diffuse_range: tuple[float, float] = light_diffuse_range  # \in [0., 1.]
        self.light_specular_range: tuple[float, float] = light_specular_range  # \in [0., 1.]
        self.light_shininess_range: tuple[int, int] = light_shininess_range  # \in [0, 1000]
        self.light_color_range: tuple[float, float] = light_color_range  # \in [0., 1.]

        self.objects: dict[int, ObjMesh] = {}
        self.objects_eval: dict[int, ObjMesh] = {}

        self._set_model_meshes()

    def __iter__(self) -> Iterator:
        item = None
        while True:
            cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c = self._get_pose_gt(item)
            img = self._get_bg_img(1 if self.scene_mode else len(obj_id))
            yield self._get_sample(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img)

    def _set_model_meshes(self):
        for obj_id in self.obj_list:
            self.objects[obj_id] = \
                RegularMesh(dtype=self.dtype, device=self.device, obj_id=int(obj_id), name=self.obj_list[obj_id])
        self.objects_eval = self.objects

    def _get_pose_gt(self, item):
        obj_id = torch.tensor([oid for oid in self.obj_list], dtype=torch.uint8, device=self.device)
        num_obj = len(obj_id)
        selected, _ = torch.multinomial(torch.ones(len(obj_id)), num_obj, replacement=False).sort()
        obj_id = obj_id[selected]

        radii = torch.tensor([self.objects[int(oid)].radius for oid in obj_id], dtype=self.dtype, device=self.device)
        # [N]
        centers = torch.stack([self.objects[int(oid)].center for oid in obj_id], dim=0)  # [N, 3(XYZ)]

        cam_K = utils.transform_3d.normalize_cam_K(self.cam_K.to(self.device)).expand(len(obj_id), -1, -1)  # [N, 3, 3]
        box2d_min = torch.linalg.inv(cam_K)[..., -1]  # [N, 3(XY1)], inv(K) @ [0., 0., 1.].T
        box2d_max = torch.linalg.solve(cam_K, torch.tensor([[self.img_render_size], [self.img_render_size], [1.]],
            dtype=self.dtype, device=self.device))[..., 0]  # [N, 3(XY1)], inv(K) @ [W, H, 1.].T
        t_depth_min, t_depth_max = self.random_t_depth_range
        box3d_size = (box2d_max - box2d_min) * t_depth_min - radii[:, None] * 2.  # [N, 3(XYZ)]
        box3d_size[..., -1] += t_depth_max - t_depth_min
        box3d_min = box2d_min * t_depth_min - centers + radii[:, None]  # [N, 3(XYZ)]

        if self.scene_mode:
            triu_indices = torch.triu_indices(num_obj, num_obj, 1)
            mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

        while True:
            gt_cam_t_m2c = torch.rand((num_obj, 3), dtype=self.dtype, device=self.device) * box3d_size + box3d_min
            if not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all():
                break
        gt_cam_R_m2c = pytorch3d.transforms.random_rotations(num_obj, dtype=self.dtype, device=self.device)  # [N, 3, 3]
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c

    def _get_bg_img(self, num):
        if self._bg_img_path is not None:
            transform = T.RandomCrop(self.img_render_size)
            idx = torch.randint(len(self._bg_img_path), [num])
            img = []
            for i in idx:
                im = utils.io.read_img_file(self._bg_img_path[i], dtype=self.dtype, device=self.device)
                im_size = min(im.shape[-2:])
                if im_size < self.img_render_size:
                    im = vF.resize(im, [self.img_render_size])
                im = transform(im)
                img.append(im)
            img = torch.cat(img, dim=0)
        else:
            img = torch.zeros(1, 3, self.img_render_size, self.img_render_size, dtype=self.dtype, device=self.device)
        return img

    def _get_sample(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, img) -> Sample:
        _, _, height, width = img.shape
        coord_2d = utils.image_2d.get_coord_2d_map(width, height, cam_K)
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]

        img, gt_coord_3d, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj = \
            self._get_features(item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img)

        if self.transform is not None:
            img = self.transform(img)  # [B, 3(RGB), H, W]

        selected = gt_vis_ratio >= self.vis_ratio_filter_threshold  # visibility threshold to select object
        bbox = gt_bbox_vis[selected]
        dzi_ratio = (torch.rand(selected.sum(), 4, dtype=self.dtype, device=self.device) * 2. - 1.) * self.max_dzi_ratio
        dzi_ratio[:, 3] = dzi_ratio[:, 2]
        bbox = utils.image_2d.get_dzi_bbox(bbox, dzi_ratio)  # [N, 4(XYWH)]
        crop_size = utils.image_2d.get_dzi_crop_size(bbox, self.bbox_zoom_out_ratio)
        crop = lambda img, out_size=self.pnp_input_size: utils.image_2d.crop_roi(img, bbox, crop_size, out_size)

        coord_2d_roi, gt_coord_3d_roi, gt_mask_vis_roi, gt_mask_obj_roi = crop([
            coord_2d[selected], gt_coord_3d[selected],
            gt_mask_vis[selected].to(dtype=self.dtype), gt_mask_obj[selected].to(dtype=self.dtype)])

        # F.interpolate doesn't support bool
        sample = Sample(
            obj_id=obj_id[selected], obj_size=obj_size[selected], cam_K=cam_K[selected],
            gt_cam_R_m2c=gt_cam_R_m2c[selected], gt_cam_t_m2c=gt_cam_t_m2c[selected],
            coord_2d_roi=coord_2d_roi, gt_coord_3d_roi=gt_coord_3d_roi,
            gt_mask_vis_roi=gt_mask_vis_roi.round().bool(), gt_mask_obj_roi=gt_mask_obj_roi.round().bool(),
            img_roi=crop(img.squeeze(), self.img_input_size), dbg_img=img if cc.debug_mode else None,
            bbox=bbox, gt_bbox_vis=gt_bbox_vis[selected], gt_bbox_obj=gt_bbox_obj[selected],
            bbox_zoom_out_ratio=self.bbox_zoom_out_ratio,
        )
        return sample

    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img):
        scene_cls = Scene if self.scene_mode else SceneBatch
        height, width = coord_2d.shape[-2:]
        scene = scene_cls(objects=self.objects, cam_K=cam_K, obj_id=obj_id, gt_cam_R_m2c=gt_cam_R_m2c,
                          gt_cam_t_m2c=gt_cam_t_m2c, width=width, height=height)
        B = 1 if self.scene_mode else len(obj_id)
        light_color = torch.rand(B, 3) * (self.light_color_range[1] - self.light_color_range[0]) \
                      + self.light_color_range[0]

        def get_light(intensity_range):
            light_intensity = torch.rand(B, 1) * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
            return light_intensity * light_color

        direction = torch.randn(B, 3)
        shininess = torch.randint(low=self.light_shininess_range[0], high=self.light_shininess_range[1] + 1, size=(B,))
        images = scene.render_scene(
            ambient=get_light(self.light_ambient_range), diffuse=get_light(self.light_diffuse_range),
            specular=get_light(self.light_specular_range), direction=direction, shininess=shininess
        )  # [B, 3(RGB), H, W]
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
        objects_info = utils.io.read_json_file(os.path.join(path_models, 'models_info.json'))
        objects_info_eval = utils.io.read_json_file(os.path.join(path_models_eval, 'models_info.json'))
        for obj_id in self.obj_list:
            self.objects[obj_id] = BOPMesh(dtype=self.dtype, device=self.device,
                                           obj_id=int(obj_id), name=self.obj_list[obj_id], is_eval=False,
                                           mesh_path=os.path.join(path_models, f'obj_{int(obj_id):0>6d}.ply'),
                                           **objects_info[str(obj_id)])
            self.objects_eval[obj_id] = BOPMesh(dtype=self.dtype, device=self.device,
                                                obj_id=int(obj_id), name=self.obj_list[obj_id], is_eval=True,
                                                mesh_path=os.path.join(path_models_eval, f'obj_{int(obj_id):0>6d}.ply'),
                                                **objects_info_eval[str(obj_id)])


class RenderedPoseBOPObjDataset(RandomPoseBOPObjDataset):
    def __init__(self, **kwargs):
        kwargs['bg_img_path'] = None
        kwargs['img_render_size'] = None
        kwargs['cam_K'] = None
        kwargs['random_t_depth_range'] = None
        super().__init__(**kwargs)
        self._scene_camera: list[dict[str, Any]] = []
        self._scene_gt: list[dict[str, Any]] = []
        self._scene_gt_info: list[dict[str, Any]] = []
        self._scene_id: list[int] = []
        self._data_path: str = os.path.join(self.path, 'test_all')
        self._data_path: list[str] = []
        self._read_scene_gt_from_BOP()

    def __len__(self) -> int:
        return len(self._scene_gt)

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
            dir_scene_camera = utils.io.read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            self._scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
            dir_scene_gt = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            self._scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
            self._scene_id += [int(key) for key in dir_scene_gt]
            self._data_path += [dir_path] * len(dir_scene_gt)
            dir_scene_gt_info = utils.io.read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
            self._scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]

    def _get_pose_gt(self, item):
        scene_gt = torch.utils.data.dataloader.default_collate(self._scene_gt[item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=self.dtype).reshape(-1, 3, 3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=self.dtype) * BOPMesh.scale
        cam_K = torch.tensor(self._scene_camera[item]['cam_K'], device=self.device)
        cam_K = utils.transform_3d.normalize_cam_K(cam_K.reshape(3, 3)).expand(len(obj_id), -1, -1)
        return cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c

    def _get_bg_img(self, item):
        img_path = os.path.join(self._data_path[item], 'rgb/{:0>6d}.png'.format(self._scene_id[item]))
        img = utils.io.read_img_file(img_path, dtype=self.dtype, device=self.device)
        return img


class BOPObjDataset(RenderedPoseBOPObjDataset):
    def __init__(self, **kwargs):
        kwargs['scene_mode'] = None
        super().__init__(**kwargs)

    def _get_features(self, item, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coord_2d, img):
        data_path = self._data_path[item]
        scene_id = self._scene_id[item]
        depth_path = os.path.join(data_path, 'depth/{:0>6d}.png'.format(scene_id))
        depth_img = utils.io.read_depth_img_file(depth_path, dtype=self.dtype, device=self.device)  # [1, 1, H, W]
        N, _, H, W = coord_2d.shape
        depth_mask = depth_img.bool()
        depth_img *= self._scene_camera[item]['depth_scale'] * BOPMesh.scale
        depth_img = torch.cat([coord_2d * depth_img, depth_img.expand(N, -1, -1, -1)], dim=1)  # [N, 3(XYZ), H, W]
        gt_coord_3d_vis = (gt_cam_R_m2c.transpose(-2, -1) @ (depth_img.reshape(N, 3, -1) - gt_cam_t_m2c[..., None])) \
                              .reshape(-1, 3, H, W) * depth_mask

        gt_mask_obj = torch.empty(len(obj_id), 1, H, W).to(self.device, dtype=torch.bool)
        gt_mask_vis = torch.empty_like(gt_mask_obj)
        for i in range(len(obj_id)):
            mask_obj_path = os.path.join(data_path, 'mask/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_obj[i] = utils.io.read_depth_img_file(mask_obj_path, dtype=self.dtype, device=self.device)
            mask_vis_path = os.path.join(data_path, 'mask_visib/{:0>6d}_{:0>6d}.png'.format(scene_id, i))
            gt_mask_vis[i] = utils.io.read_depth_img_file(mask_vis_path, dtype=self.dtype, device=self.device)

        # gt_mask = gt_mask_obj.any(dim=0)[None]

        def cvt_bbox(tensor_list_4):
            bbox = torch.stack(tensor_list_4, dim=-1).to(self.device, dtype=self.dtype)
            bbox[:, :2] += bbox[:, 2:] * .5
            return bbox

        scene_gt_info = torch.utils.data.dataloader.default_collate(self._scene_gt_info[item])
        gt_bbox_vis = cvt_bbox(scene_gt_info['bbox_visib'])
        gt_bbox_obj = cvt_bbox(scene_gt_info['bbox_obj'])
        gt_vis_ratio = scene_gt_info['visib_fract'].to(self.device, dtype=self.dtype)
        return img, gt_coord_3d_vis, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj
