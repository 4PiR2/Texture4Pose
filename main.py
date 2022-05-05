import time

import cv2
import numpy as np
import pytorch3d
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from dataloader.BOPDataset import BOPDataset
from dataloader.Sample import Sample
from models.ConvPnPNet import ConvPnPNet
from models.Loss import Loss
from models.gdrn import GDRN
from models.resnet_backbone import ResNetBackboneNet, resnet_spec
from models.rot_head import RotWithRegionHead
from utils.const import lmo_objects, device, debug_mode

if __name__ == '__main__':
    block_type, layers, channels, _ = resnet_spec[34]
    backbone_net = ResNetBackboneNet(block_type, layers, in_channel=3)
    rot_head_net = RotWithRegionHead(channels[-1], num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                                     num_regions=64)
    pnp_net = ConvPnPNet(nIn=5)
    model = GDRN(backbone_net, rot_head_net, pnp_net).to(device)
    model.load_pretrain('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')

    composed = None  # T.Compose([T.RandomGrayscale(p=0.1)])
    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', transform=composed, device=device)

    # train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=Sample.collate)
    # model = ConvPnPNet(nIn=5).to(device)
    # optimizer = Adam(model.parameters(), lr=1e-3)
    #
    # ts = time.time()
    # L = Loss(dataset)
    # for epoch in range(100):
    #     for sample in train_dataloader:
    #         te = time.time()
    #         R, t, t_site = model(sample)
    #         loss = L(sample, R, t_site)
    #         angle, dist = sample.relative_angle(R), sample.relative_dist(t)
    #         print(float(loss), float(angle.mean()), float(dist.mean()))
    #         optimizer.zero_grad()  # clear gradients for next train
    #         loss.backward()  # backpropagation, compute gradients
    #         optimizer.step()  # apply gradients
    #         print(te - ts)
    #         ts = te
    #     a = 0

    scene_id_test = 3
    sample = dataset[scene_id_test]

    model.eval()
    R, t, t_site = model(sample)
    angle, dist = sample.relative_angle(R), sample.relative_dist(t)
    angle *= 180. / torch.pi
    dist *= 100.

    if debug_mode:
        sample.visualize()
    R, t = sample.sanity_check()
    angle, dist = sample.relative_angle(R), sample.relative_dist(t)
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
