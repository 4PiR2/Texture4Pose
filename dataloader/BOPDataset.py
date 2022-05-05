import os.path
from typing import Any

import cv2
import torch
import torchvision.transforms.functional as vF
from torch.utils.data import Dataset

from dataloader.ObjMesh import ObjMesh
from dataloader.Sample import Sample
from dataloader.Scene import Scene
from utils.const import debug_mode, pnp_input_size, img_input_size, gdr_mode
from utils.io import read_json_file, parse_device
from utils.transform import calculate_bbox_crop, t_to_t_site


class BOPDataset(Dataset):
    def __init__(self, obj_list, path, transform=None, read_scene_from_bop=True, device=None):
        self.obj_list: list[int] = obj_list
        self.path: str = path
        self.transform: Any = transform
        self.device: torch.device = parse_device(device)

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

        self.data_path = os.path.join(path, 'test_all')
        if read_scene_from_bop:
            for dir in os.listdir(self.data_path):
                dir_path = os.path.join(self.data_path, dir)
                dir_scene_camera = read_json_file(os.path.join(dir_path, 'scene_camera.json'))
                for key in dir_scene_camera:
                    self.scene_camera.append(dir_scene_camera[key])
                dir_scene_gt = read_json_file(os.path.join(dir_path, 'scene_gt.json'))
                for key in dir_scene_gt:
                    self.scene_gt.append(dir_scene_gt[key])

    def __len__(self):
        return len(self.scene_gt)

    def __getitem__(self, item) -> Sample:
        obj_id = []
        gt_cam_R_m2c = []
        gt_cam_t_m2c = []
        for pose in self.scene_gt[item]:
            obj_id.append(pose['obj_id'])
            gt_cam_R_m2c.append(torch.tensor(pose['cam_R_m2c'], device=self.device).reshape(3, 3))
            gt_cam_t_m2c.append(torch.tensor(pose['cam_t_m2c'], device=self.device) * ObjMesh.scale)

        obj_id = torch.Tensor(obj_id).to(self.device, dtype=torch.uint8)
        gt_cam_R_m2c = torch.stack(gt_cam_R_m2c, dim=0)
        gt_cam_t_m2c = torch.stack(gt_cam_t_m2c, dim=0)
        cam_K = torch.tensor(self.scene_camera[item]['cam_K'], device=self.device).reshape(3, 3)
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]

        scene = Scene(objects=self.objects, cam_K=cam_K, obj_id=obj_id, cam_R_m2c=gt_cam_R_m2c,
                      cam_t_m2c=gt_cam_t_m2c, width=640, height=480, device=self.device)

        bg_path = os.path.join(self.data_path, '{:0>6d}/rgb/{:0>6d}.png'.format(2, item))
        bg = torch.tensor(cv2.imread(bg_path, cv2.IMREAD_COLOR)[:, :, ::-1].copy(), device=self.device) \
                 .permute(2, 0, 1)[None] / 255.  # [1, 3(RGB), H, W] \in [0, 1]

        a = torch.rand(1) * .5 + .5
        d = torch.rand(1) * (1. - a)
        s = 1. - (a + d)
        ambient = a.expand(3)[None]
        diffuse = d.expand(3)[None]
        specular = s.expand(3)[None]
        direction = torch.randn(3)[None]
        shininess = torch.randint(low=40, high=80, size=(1,))  # shininess: 0-1000
        image = scene.render_scene_mesh(ambient=ambient, diffuse=diffuse, specular=specular, direction=direction,
                                        shininess=shininess, bg=bg)
        # [1, 3(RGB), H, W]

        if self.transform is not None:
            image = self.transform(image)  # [1, 3(RGB), H, W]

        selected = scene.gt_vis_ratio >= .5  # visibility threshold to select object
        bbox = scene.gt_bbox_vis[selected]  # [N, 4(XYWH)]
        if gdr_mode:
            bbox[:, 2:] *= 1.5
        crop_size, pad_size, x0, y0 = calculate_bbox_crop(bbox)

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
                        coor2d=crop(scene.coor2d), gt_coor3d=crop(scene.gt_coor3d_obj[selected]),
                        gt_mask_vis=crop(scene.gt_mask_vis[selected].to(dtype=torch.uint8)).bool(),
                        gt_mask_obj=crop(scene.gt_mask_obj[selected].to(dtype=torch.uint8)).bool(),
                        img=crop(image[0], img_input_size),
                        dbg_img=image if debug_mode else None,
                        bbox=bbox,
                        )
        return result
