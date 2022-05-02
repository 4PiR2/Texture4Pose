import time

import cv2
import numpy as np
import pytorch3d
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from dataloader.BOPDataset import BOPDataset
from dataloader.Sample import Sample
from models.ConvPnPNet import ConvPnPNet
from models.Loss import Loss
from utils.const import lmo_objects, device, debug_mode

if __name__ == '__main__':
    composed = None  # T.Compose([T.RandomGrayscale(p=0.1)])

    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', transform=composed, device=device)

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=Sample.collate)
    model = ConvPnPNet(nIn=5).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    ts = time.time()
    # for sample in train_dataloader:
    #     te = time.time()
    #     rot, t = model(torch.cat([sample.gt_coor3d, sample.coor2d], dim=1))
    #     rot = pytorch3d.transforms.rotation_6d_to_matrix(rot)
    #     loss, _, dist = sample.eval_pred(rot, t)
    #     print(float(loss))
    #     optimizer.zero_grad()  # clear gradients for next train
    #     loss.backward()  # backpropagation, compute gradients
    #     optimizer.step()  # apply gradients
    #     print(te - ts)
    #     ts = te

    scene_id_test = 3
    result = dataset[scene_id_test]
    if debug_mode:
        result.visualize()
    R, t = result.sanity_check()
    loss = Loss(dataset, result)
    angle, dist = loss.relative_angle(R), loss.relative_dist(t)
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
