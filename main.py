import time

import cv2
import numpy as np
import torch
from pytorch3d.transforms import so3_relative_angle
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from dataloader.BOPDataset import BOPDataset
from dataloader.Sample import Sample
from utils.const import lmo_objects, device

if __name__ == '__main__':
    composed = None  # T.Compose([T.RandomGrayscale(p=0.1)])

    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', transform=composed, device=device)

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=Sample.collate)
    # ts = time.time()
    # for i in train_dataloader:
    #     te = time.time()
    #     print(te - ts)
    #     ts = te

    scene_id_test = 3
    result = dataset[scene_id_test]
    result.visualize()

    for i in range(1, 2):
        mask = result['gt_mask_vis'][i].squeeze()
        x = result['gt_coor3d'][i].permute(1, 2, 0)[mask]
        y = result['coor2d'][i].permute(1, 2, 0)[mask]

        gt_K = result['cam_K']
        gt_R = result['gt_cam_R_m2c'][i]
        gt_t = result['gt_cam_t_m2c'][i]

        _, pred_R_exp, pred_t, _ = cv2.solvePnPRansac(x.cpu().numpy(), y.cpu().numpy(), np.eye(3), None)
        pred_R, _ = cv2.Rodrigues(pred_R_exp)
        pred_R, pred_t = torch.Tensor(pred_R).to(device), torch.Tensor(pred_t).to(device).flatten()

        angle = so3_relative_angle(pred_R[None], gt_R[None])
        dist = torch.norm(pred_t - gt_t)

        gt_proj = (x @ gt_R.T + gt_t) @ gt_K.T
        gt_proj = gt_proj[:, :2] / gt_proj[:, 2:]

        pred_proj = (x @ pred_R.T + pred_t) @ gt_K.T
        pred_proj = pred_proj[:, :2] / pred_proj[:, 2:]

        a = 0


# def poses_from_random(dataset, num_obj):
#     device = dataset.device
#     euler_angles = (2. * torch.pi) * torch.rand((num_obj, 3), device=device)
#     Rs = euler_angles_to_matrix(euler_angles, 'ZYX')
#
#     objects = dataset.objects
#     selected_obj = [obj_id for obj_id in objects]
#     random.shuffle(selected_obj)
#     selected_obj = selected_obj[:num_obj]
#     selected_obj.sort()
#     radii = torch.tensor([objects[obj_id].radius for obj_id in selected_obj], device=device)
#     centers = torch.stack([objects[obj_id].center for obj_id in selected_obj], dim=0)
#     triu_indices = torch.triu_indices(num_obj, num_obj, 1)
#     mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]
#
#     flag = False
#     while not flag:
#         positions = torch.rand((num_obj, 3), device=device)\
#                     * torch.tensor((.5, .5, .5), device=device) + torch.tensor((-.25, -.25, 1.), device=device)
#         flag = (F.pdist(positions) >= mdist).all()
#     positions -= centers
#
#     poses = []
#     for i in range(num_obj):
#         poses.append({'obj_id': selected_obj[i], 'cam_R_m2c': Rs[i], 'cam_t_m2c': positions[i]})
#     return poses
