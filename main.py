import gc
import pickle
import time

import cv2
import numpy as np
import pytorch3d
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from dataloader.pose_dataset import BOPObjDataset, RenderedPoseBOPObjDataset, RandomPoseBOPObjDataset, RandomPoseRegularObjDataset
from dataloader.sample import Sample
from engine.trainer import LitModel
from utils.const import lmo_objects, device, debug_mode, lm_objects, lm13_objects, gdr_mode, regular_objects

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    # dataset = RandomPoseRegularObjDataset(obj_list=regular_objects, scene_mode=True, device=device, bg_img_path='/data/coco/train2017')
    # dataset = RandomPoseBOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', scene_mode=False, device=device, bg_img_path='/data/coco/train2017')
    # dataset = RenderedPoseBOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', scene_mode=True, device=device)
    # dataset = BOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', device=device)
    # for s in dataset:
    #     s.visualize()
    #     a = 0

    test_objects = {101: 'sphere'}

    #dataset = RandomPoseRegularObjDataset(obj_list=test_objects, scene_mode=False, device=device, bg_img_path='/data/coco/train2017')
    dataset = BOPObjDataset(obj_list={1: 'ape'}, path='data/BOP/lm', device=device)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=Sample.collate)

    model = LitModel(dataset.objects)
    model.load_pretrain('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')
    model = model.to(device)

    # trainer = Trainer(
    #     accelerator='auto',
    #     devices=1 if torch.cuda.is_available() else None,
    #     max_epochs=3,
    #     callbacks=[TQDMProgressBar(refresh_rate=20)],
    #     default_root_dir='outputs',
    # )
    #
    # trainer.fit(model, dataloader)

    composed = None  # T.Compose([T.RandomGrayscale(p=0.1)])

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
    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_errors.pkl', 'rb') as f:
        gdr_errors = pickle.load(f)['ape']
    N = len(gdr_results)
    results = []
    model.eval()
    with torch.no_grad():
        i = 0
        for sample, pic_path in zip(dataloader, gdr_results):
            id = int(pic_path.split('/')[-1].split('.')[0])
            if i < id:
                continue
            gdr_result = gdr_results[pic_path]
            sample.img = torch.load(f'/home/user/Desktop/x/data/{id}.pth').flip(dims=[1])
            R, t, *_ = model(sample)
            re, te = sample.relative_angle(R, degree=True), sample.relative_dist(t, cm=True)
            ad = sample.add_score(dataset.objects_eval, R, t)
            proj = sample.proj_dist(dataset.objects_eval, R, t)
            gdr_R = torch.tensor(gdr_result['R'], device=device)[None]
            gdr_t = torch.tensor(gdr_result['t'], device=device)[None]
            gdr_re = sample.relative_angle(gdr_R, degree=True)
            gdr_te = sample.relative_dist(gdr_t, cm=True)
            gdr_ad = sample.add_score(dataset.objects_eval, gdr_R, gdr_t)
            gdr_proj = sample.proj_dist(dataset.objects_eval, gdr_R, gdr_t)

            result = [[float(re), float(te), float(ad), float(proj)],
                      [float(gdr_re), float(gdr_te), float(gdr_ad), float(gdr_proj)],
                      [gdr_errors['re'][id], gdr_errors['te'][id] * 100, gdr_errors['ad'][id] / dataset.objects_eval[1].diameter, gdr_errors['proj'][id]]]
            result = np.array(result)
            results.append(result)
            # gc.collect()
            # torch.cuda.empty_cache()
            i += 1
    results = np.array(results)

    R, t = sample.sanity_check()
    re, te = sample.relative_angle(R), sample.relative_dist(t)
    a = 0
