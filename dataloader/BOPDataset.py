import os.path

import cv2
import torch
import torch.nn.functional as F

from dataloader.BOPObjSet import BOPObjSet
from dataloader.ObjMesh import ObjMesh
from dataloader.Scene import Scene
from utils.io import read_json_file
from utils.const import debug_mode


class BOPDataset:
    def __init__(self, obj_list, path, device=None):
        self.device = device if device is not None else 'cpu'
        self.object_set = BOPObjSet(obj_list=obj_list, path=path, device=self.device)
        self.object_set.load_meshes(flag=True)
        self.data_path = os.path.join(path, 'test_all')
        self.scene_camera = []
        self.scene_gt = []
        for dir in os.listdir(self.data_path):
            dir_path = os.path.join(self.data_path, dir)
            dir_scene_camera = read_json_file(os.path.join(dir_path, 'scene_camera.json'))
            for key in dir_scene_camera:
                self.scene_camera.append(dir_scene_camera[key])
            dir_scene_gt = read_json_file(os.path.join(dir_path, 'scene_gt.json'))
            for key in dir_scene_gt:
                self.scene_gt.append(dir_scene_gt[key])
        width, height = float(Scene.width), float(Scene.height)
        if debug_mode:
            width, height = 1., 1.
        self.coor2d = torch.stack(torch.meshgrid(torch.arange(0., width, step=width/Scene.width),
                                                 torch.arange(0., height, step=height/Scene.height),
                                                 indexing='xy'), dim=0).to(self.device)

    def __len__(self):
        return len(self.scene_gt)

    def __getitem__(self, item):
        gt_obj_id = []
        gt_cam_R_m2c = []
        gt_cam_t_m2c = []
        for pose in self.scene_gt[item]:
            gt_obj_id.append(pose['obj_id'])
            gt_cam_R_m2c.append(torch.tensor(pose['cam_R_m2c'], device=self.device).reshape(3, 3))
            gt_cam_t_m2c.append(torch.tensor(pose['cam_t_m2c'], device=self.device) * ObjMesh.scale)

        gt_poses = {'obj_id': torch.Tensor(gt_obj_id).to(self.device),
                    'cam_R_m2c': torch.stack(gt_cam_R_m2c, dim=0),
                    'cam_t_m2c': torch.stack(gt_cam_t_m2c, dim=0)}

        scene = Scene(obj_set=self.object_set, gt_poses=gt_poses, **self.scene_camera[item])
        transformed_meshes = scene.get_transformed_meshes()
        mask, (_, gt_mask_vis, bbox_vis), (gt_coor3d_obj, gt_mask_obj, _) =\
            scene.render_scene_mesh_gt(transformed_meshes=transformed_meshes)
        img = scene.render_scene_mesh(transformed_meshes=transformed_meshes)

        gt_coor3d_obj = gt_coor3d_obj.permute(0, 3, 1, 2)
        mask = mask[None, None]
        gt_mask_vis = gt_mask_vis[:, None].to(dtype=torch.uint8)  # F.interpolate doesn't support bool
        gt_mask_obj = gt_mask_obj[:, None].to(dtype=torch.uint8)
        img = img.permute(0, 3, 1, 2)

        bg_path = os.path.join(self.data_path, '{:0>6d}/rgb/{:0>6d}.png'.format(2, item))
        bg = torch.tensor(cv2.imread(bg_path, cv2.IMREAD_COLOR)[:, :, ::-1].copy(), device=self.device) / 255.
        bg = bg.permute(2, 0, 1)[None]  # [1, 3(RGB), H, W] \in [0, 1]
        img = img * mask + bg * ~mask

        def crop(bbox, output_size, *imgs):
            # bbox [N, 4(XYWH)]
            # output_size int, output is [N, C, output_size x output_size] image
            # imgs: list of [N, C, H, W] or [C, H, W] images
            crop_size, _ = bbox[:, 2:].max(dim=-1)
            max_crop_size = int(crop_size.max())
            pad_size = (max_crop_size + 1) // 2
            x0, y0 = (bbox[:, :2].T - crop_size * .5).int() + pad_size
            crop_size = crop_size.int()
            x1, y1 = x0 + crop_size, y0 + crop_size

            cropped_imgs = []
            for img in imgs:
                padded_img = torch.zeros([*img.shape[:-2], img.shape[-2]+max_crop_size, img.shape[-1]+max_crop_size],
                                         dtype=img.dtype, device=img.device)
                padded_img[..., pad_size:pad_size+img.shape[-2], pad_size:pad_size+img.shape[-1]] = img
                c_imgs = []
                for i in range(len(bbox)):
                    if img.dim() > 3:  # img [N, C, H, W]
                        c_img = padded_img[i:i+1, ..., y0[i]:y1[i], x0[i]:x1[i]]
                    else:  # img [C, H, W]
                        c_img = padded_img[None, ..., y0[i]:y1[i], x0[i]:x1[i]]
                    if padded_img.dtype in [torch.uint8]:
                        c_img = F.interpolate(c_img, size=output_size, mode='nearest')
                    else:
                        c_img = F.interpolate(c_img, size=output_size, mode='bilinear', align_corners=True)
                    c_imgs.append(c_img)  # c_img [1, C, H, W]
                cropped_imgs.append(torch.cat(c_imgs, dim=0))  # [N, C, H, W]
            return cropped_imgs

        result = {'gt_obj_id': gt_poses['obj_id'],
                  'gt_cam_R_m2c': gt_poses['cam_R_m2c'],
                  'gt_cam_t_m2c': gt_poses['cam_t_m2c']}
        result['gt_coor2d'], result['gt_coor3d'], mask_vis_c, mask_obj_c, *result['imgs']\
            = crop(bbox_vis, 64, self.coor2d, gt_coor3d_obj, gt_mask_vis, gt_mask_obj, *img)

        result['gt_mask_vis'], result['gt_mask_obj'] = mask_vis_c.bool(), mask_obj_c.bool()
        result['dbg_img'], result['dbg_bbox'] = img, bbox_vis
        return result
