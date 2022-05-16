import gc
import pickle
import time

import cv2
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, so3_rotation_angle, \
    matrix_to_euler_angles, random_rotations
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from matplotlib import pyplot as plt

from dataloader.BOPDataset import BOPDataset
from dataloader.Sample import Sample
from models.ConvPnPNet import ConvPnPNet
from models.Loss import Loss
from models.gdrn import GDRN
from models.resnet_backbone import ResNetBackboneNet, resnet_spec
from models.rot_head import RotWithRegionHead
from utils.const import lmo_objects, device, debug_mode, lm_objects, lm13_objects, gdr_mode

if __name__ == '__main__':
    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', render_mode=True, lmo_mode=True, device=device, read_scene_from_bop=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=Sample.collate)
    for x in dataloader:
        x.visualize()
        a = 0


    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', render_mode=True, lmo_mode=True, device=device)
    for sample in dataset:
        sample.visualize()

    block_type, layers, channels, _ = resnet_spec[34]
    backbone_net = ResNetBackboneNet(block_type, layers, in_channel=3)
    rot_head_net = RotWithRegionHead(channels[-1], num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                                     num_regions=64)
    pnp_net = ConvPnPNet(nIn=5)
    model = GDRN(backbone_net, rot_head_net, pnp_net).to(device)
    model.load_pretrain('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')

    composed = None  # T.Compose([T.RandomGrayscale(p=0.1)])
    test_objects = {1: 'ape'}
    dataset = BOPDataset(obj_list=test_objects, path='data/BOP/lm', render_mode=False, lmo_mode=False, device=device)

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

    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_preds.pkl', 'rb') as f:
        gdr_results = pickle.load(f)['ape']
    model.eval()
    N = len(gdr_results)
    results = []
    with torch.no_grad():
        i = 0
        for pic_path in gdr_results:
            id = int(pic_path.split('/')[-1].split('.')[0])
            gdr_result = gdr_results[pic_path]
            sample = dataset[id]
            # sample.img = torch.load(f'/home/user/Desktop/x/data/{i}.pth').flip(dims=[1])
            R, t, *_ = model(sample)
            angle, dist = sample.relative_angle(R), sample.relative_dist(t)
            add = sample.add_score(dataset.objects_eval, R, t)
            gdr_R = torch.tensor(gdr_result['R'], device=device)[None]
            gdr_t = torch.tensor(gdr_result['t'], device=device)[None]
            gdr_angle = sample.relative_angle(gdr_R)
            gdr_dist = sample.relative_dist(gdr_t)
            gdr_add = sample.add_score(dataset.objects_eval, gdr_R, gdr_t)

            result = [float(angle), float(gdr_angle), float(dist), float(gdr_dist), float(add), float(gdr_add)]
            results.append(result)
            # gc.collect()
            # torch.cuda.empty_cache()
            i += 1
    results = np.array(results)

    R, t = sample.sanity_check()
    angle, dist = sample.relative_angle(R), sample.relative_dist(t)
    a = 0
