import os.path
from typing import Any

import torch
import torchvision.transforms.functional as vF
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataloader.ObjMesh import ObjMesh
from dataloader.Sample import Sample
from dataloader.Scene import Scene
from utils.const import debug_mode, pnp_input_size, img_input_size, gdr_mode
from utils.io import read_json_file, parse_device, read_img_file, read_depth_img_file
from utils.transform import calc_bbox_crop, t_to_t_site, get_coor2d_map


class BOPDataset(Dataset):
    def __init__(self, obj_list, path, transform=None, read_scene_from_bop=True,
                 render_mode=True, lmo_mode=True, device=None):
        self.obj_list: list[int] = obj_list
        self.path: str = path
        self.transform: Any = transform
        self.device: torch.device = parse_device(device)
        self.render_mode: bool = render_mode
        self.lmo_mode: bool = lmo_mode

        path_models = os.path.join(path, 'models')
        path_models_eval = os.path.join(path, 'models_eval')
        objects_info = read_json_file(os.path.join(path_models, 'models_info.json'))
        objects_info_eval = read_json_file(os.path.join(path_models_eval, 'models_info.json'))

        self.objects: dict[int, ObjMesh] = {}
        self.objects_eval: dict[int, ObjMesh] = {}
        for obj_id in obj_list:
            self.objects[obj_id] = ObjMesh(device=self.device, obj_id=int(obj_id), name=obj_list[obj_id], is_eval=False,
                                           mesh_path=os.path.join(path_models, f'obj_{int(obj_id):0>6d}.ply'),
                                           **objects_info[str(obj_id)])
            self.objects_eval[obj_id] = ObjMesh(device=self.device, obj_id=int(obj_id), name=obj_list[obj_id],
                                                is_eval=True,
                                                mesh_path=os.path.join(path_models_eval, f'obj_{int(obj_id):0>6d}.ply'),
                                                **objects_info_eval[str(obj_id)])

        self.scene_camera: list[dict[str, Any]] = []
        self.scene_gt: list[dict[str, Any]] = []
        self.scene_gt_info: list[dict[str, Any]] = []
        self.scene_id: list[int] = []

        self.data_path: str = os.path.join(path, 'test_all')
        if read_scene_from_bop:
            for dir in os.listdir(self.data_path):
                if not dir.startswith('0'):
                    continue
                if not self.lmo_mode and int(dir) not in self.obj_list:
                    continue
                dir_path = os.path.join(self.data_path, dir)
                dir_scene_camera = read_json_file(os.path.join(dir_path, 'scene_camera.json'))
                self.scene_camera += [dir_scene_camera[key] for key in dir_scene_camera]
                dir_scene_gt = read_json_file(os.path.join(dir_path, 'scene_gt.json'))
                self.scene_gt += [dir_scene_gt[key] for key in dir_scene_gt]
                self.scene_id += [int(key) for key in dir_scene_gt]
                dir_scene_gt_info = read_json_file(os.path.join(dir_path, 'scene_gt_info.json'))
                self.scene_gt_info += [dir_scene_gt_info[key] for key in dir_scene_gt_info]

    def __len__(self):
        return len(self.scene_gt)

    def __getitem__(self, item) -> Sample:
        scene_gt = default_collate(self.scene_gt[item])
        obj_id = scene_gt['obj_id'].to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(scene_gt['cam_R_m2c'], dim=-1).to(self.device, dtype=torch.float).reshape(-1, 3, 3)
        gt_cam_t_m2c = torch.stack(scene_gt['cam_t_m2c'], dim=-1).to(self.device, dtype=torch.float) * ObjMesh.scale

        cam_K = torch.tensor(self.scene_camera[item]['cam_K'], device=self.device).reshape(3, 3)
        cam_K /= float(cam_K[-1, -1])
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]

        img_path = os.path.join(self.data_path, '{:0>6d}/rgb/{:0>6d}.png'.format(
            2 if self.lmo_mode else obj_id[0], self.scene_id[item]))
        img = read_img_file(img_path, device=self.device)
        _, _, height, width = img.shape
        coor2d = get_coor2d_map(width, height, cam_K)

        if self.render_mode:
            rendered_img, gt_coor3d, gt_mask, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj = \
                self.get_item_by_rendering(cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, width, height)
            img = rendered_img * gt_mask + img * ~gt_mask
        else:
            gt_coor3d, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj = \
                self.get_item_by_loading(item, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coor2d)

        if self.transform is not None:
            img = self.transform(img)  # [1, 3(RGB), H, W]

        selected = gt_vis_ratio >= .5  # visibility threshold to select object
        bbox = gt_bbox_vis[selected]  # [N, 4(XYWH)]
        crop_size, pad_size, x0, y0 = calc_bbox_crop(bbox)

        def crop(img, out_size=pnp_input_size):
            # [N, C, H, W] or [C, H, W]
            padded_img = vF.pad(img, padding=pad_size)
            c_imgs = [vF.resized_crop((padded_img[i] if img.dim() > 3 else padded_img)[None],
                                      y0[i], x0[i], crop_size[i], crop_size[i], out_size) for i in
                      range(len(bbox))]
            # [1, C, H, W]
            return torch.cat(c_imgs, dim=0)  # [N, C, H, W]

        # F.interpolate doesn't support bool
        result = Sample(obj_id=obj_id[selected], obj_size=obj_size[selected], cam_K=cam_K,
                        gt_cam_R_m2c=gt_cam_R_m2c[selected], gt_cam_t_m2c=gt_cam_t_m2c[selected],
                        gt_cam_t_m2c_site=t_to_t_site(gt_cam_t_m2c[selected], bbox, pnp_input_size / crop_size, cam_K),
                        coor2d=crop(coor2d), gt_coor3d=crop(gt_coor3d[selected]),
                        gt_mask_vis=crop(gt_mask_vis[selected].to(dtype=torch.uint8)).bool(),
                        gt_mask_obj=crop(gt_mask_obj[selected].to(dtype=torch.uint8)).bool(),
                        img=crop(img[0], img_input_size),
                        dbg_img=img if debug_mode else None,
                        bbox=bbox,
                        )
        return result

    def get_item_by_rendering(self, cam_K, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, width=512, height=512):
        scene = Scene(objects=self.objects, cam_K=cam_K, obj_id=obj_id, gt_cam_R_m2c=gt_cam_R_m2c,
                      gt_cam_t_m2c=gt_cam_t_m2c, width=width, height=height, device=self.device)

        a = torch.rand(1) * .5 + .5
        d = torch.rand(1) * (1. - a)
        s = 1. - (a + d)
        ambient = a.expand(3)[None]
        diffuse = d.expand(3)[None]
        specular = s.expand(3)[None]
        direction = torch.randn(3)[None]
        shininess = torch.randint(low=40, high=80, size=(1,))  # shininess: 0-1000
        img = scene.render_scene_mesh(ambient=ambient, diffuse=diffuse, specular=specular, direction=direction,
                                      shininess=shininess)
        # [1, 3(RGB), H, W]
        return img, scene.gt_coor3d_obj, scene.gt_mask, scene.gt_vis_ratio, \
               scene.gt_mask_vis, scene.gt_mask_obj, scene.gt_bbox_vis, scene.gt_bbox_obj

    def get_item_by_loading(self, item, obj_id, gt_cam_R_m2c, gt_cam_t_m2c, coor2d):
        scene_id = self.scene_id[item]
        o_id = 2 if self.lmo_mode else obj_id[0]
        depth_path = os.path.join(self.data_path, '{:0>6d}/depth/{:0>6d}.png'.format(o_id, scene_id))
        depth_img = read_depth_img_file(depth_path, self.device)  # [1, 1, H, W]
        H, W = depth_img.shape[-2:]
        depth_mask = depth_img != 0.
        depth_img *= self.scene_camera[item]['depth_scale'] * ObjMesh.scale
        depth_img = torch.cat([coor2d * depth_img, depth_img], dim=1)  # [1, 3(XYZ), H, W]
        gt_coor3d_vis = (gt_cam_R_m2c.transpose(-2, -1) @ (depth_img.reshape(1, 3, -1) - gt_cam_t_m2c[..., None]))\
            .reshape(-1, 3, H, W) * depth_mask

        gt_mask_obj = torch.empty(len(obj_id), 1, H, W).to(self.device, dtype=torch.bool)
        gt_mask_vis = torch.empty_like(gt_mask_obj)
        for i in range(len(obj_id)):
            mask_obj_path = os.path.join(self.data_path, '{:0>6d}/mask/{:0>6d}_{:0>6d}.png'.format(o_id, scene_id, i))
            gt_mask_obj[i] = read_depth_img_file(mask_obj_path)
            mask_vis_path = os.path.join(self.data_path,
                                         '{:0>6d}/mask_visib/{:0>6d}_{:0>6d}.png'.format(o_id, scene_id, i))
            gt_mask_vis[i] = read_depth_img_file(mask_vis_path)
        # gt_mask = gt_mask_obj.any(dim=0)[None]

        def cvt_bbox(tensor_list_4):
            bbox = torch.stack(tensor_list_4, dim=-1).to(self.device, dtype=torch.float)
            bbox[:, :2] += bbox[:, 2:] * .5
            return bbox

        scene_gt_info = default_collate(self.scene_gt_info[item])
        gt_bbox_vis = cvt_bbox(scene_gt_info['bbox_visib'])
        gt_bbox_obj = cvt_bbox(scene_gt_info['bbox_obj'])
        gt_vis_ratio = scene_gt_info['visib_fract'].to(self.device, dtype=torch.float)
        return gt_coor3d_vis, gt_vis_ratio, gt_mask_vis, gt_mask_obj, gt_bbox_vis, gt_bbox_obj
