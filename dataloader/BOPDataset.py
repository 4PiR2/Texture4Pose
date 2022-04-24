import os.path

import cv2
import torch

from dataloader.BOPObjSet import BOPObjSet
from dataloader.ObjMesh import ObjMesh
from dataloader.Scene import Scene
from utils.io import read_json_file


class BOPDataset:
    def __init__(self, obj_list, path, device=None):
        self.device = device if device is not None else 'cpu'
        self.object_set = BOPObjSet(obj_list=obj_list, path=path, device=self.device)
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

        gt_obj_id = torch.Tensor(gt_obj_id).to(self.device).int()
        gt_cam_R_m2c = torch.stack(gt_cam_R_m2c, dim=0)
        gt_cam_t_m2c = torch.stack(gt_cam_t_m2c, dim=0)

        cam_K = torch.tensor(self.scene_camera[item]['cam_K'], device=self.device).reshape(3, 3)
        scene = Scene(obj_set=self.object_set, cam_K=cam_K, obj_id=gt_obj_id, cam_R_m2c=gt_cam_R_m2c,
                      cam_t_m2c=gt_cam_t_m2c, width=640, height=480)
        
        bg_path = os.path.join(self.data_path, '{:0>6d}/rgb/{:0>6d}.png'.format(2, item))
        bg = torch.tensor(cv2.imread(bg_path, cv2.IMREAD_COLOR)[:, :, ::-1].copy(), device=self.device)\
                 .permute(2, 0, 1)[None] / 255.  # [1, 3(RGB), H, W] \in [0, 1]
        img = scene.render_scene_mesh(bg=bg)
        result = scene.get_data(scene.gt_bbox_vis, 64)
        return result
