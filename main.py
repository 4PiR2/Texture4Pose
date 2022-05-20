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


def main():
    test_objects = {101: 'sphere'}
    composed = T.Compose([T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=0.)])

    dataset = RandomPoseRegularObjDataset(obj_list=test_objects, scene_mode=False, transform=composed,
                                          bg_img_path='/data/coco/train2017', device=device)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=Sample.collate)

    model = LitModel(dataset.objects)
    model.load_pretrain('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')
    model = model.to(device)

    trainer = Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        default_root_dir='outputs',
    )

    ckpt_path = f'outputs/lightning_logs/version_{12}/checkpoints/epoch={1}-step={49}.ckpt'
    # ckpt_path = None
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader, ckpt_path=ckpt_path)


def data_loading_test():
    # dataset = RandomPoseRegularObjDataset(obj_list=regular_objects, scene_mode=True, device=device, bg_img_path='/data/coco/train2017')
    dataset = RandomPoseBOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', scene_mode=False, device=device, bg_img_path='/data/coco/train2017')
    # dataset = RenderedPoseBOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', scene_mode=True, device=device)
    # dataset = BOPObjDataset(obj_list=lmo_objects, path='data/BOP/lmo', device=device)
    for s in dataset:
        s.visualize()
        a = 0


def numerical_check(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=Sample.collate)
    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_preds.pkl', 'rb') as f:
        gdr_results = pickle.load(f)['ape']
    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_errors.pkl', 'rb') as f:
        gdr_errors = pickle.load(f)['ape']
    results = []
    model.eval()
    with torch.no_grad():
        i = 0
        for sample, pic_path in zip(dataloader, gdr_results):
            id = int(pic_path.split('/')[-1].split('.')[0])
            if i < id:
                continue
            gdr_result = gdr_results[pic_path]
            sample.img = torch.load(f'/data/x/data/{id}.pth').flip(dims=[1])
            # R, t = sample.sanity_check()
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
    return results


if __name__ == '__main__':
    main()
