import gc
import os
import pickle
import time

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataloader.pose_dataset import BOPObjDataset, RenderedPoseBOPObjDataset, RandomPoseBOPObjDataset, \
    RandomPoseRegularObjDataset, DatasetWrapper
from dataloader.sample import Sample
from engine.trainer import LitModel
from utils.config import Config
import config.const as cc


def main():
    def setup(args=None) -> Config:
        """Create configs and perform basic setups."""
        cfg = Config.fromfile('config/pipeline.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg = setup()
    # data_loading_test(cfg)

    # state_dict = torch.load('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')['model']
    # for key in state_dict:
    #     show_hist(state_dict[key], title=key, bins=100)
    #     a = 0

    test_objects = {101: 'sphere'}
    composed = T.Compose([T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=0.)])

    dataset = RandomPoseRegularObjDataset(obj_list=test_objects, scene_mode=False, transform=composed,
                                          bg_img_path='/data/coco/train2017', device=cfg.device)
    dataloader_train = DataLoader(DatasetWrapper(dataset, 10000), batch_size=16, drop_last=True, collate_fn=Sample.collate)
    dataloader_val = DataLoader(DatasetWrapper(dataset, 100), batch_size=16, collate_fn=Sample.collate)

    model = LitModel(dataset.objects)
    model.load_pretrain('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth')
    model = model.to(cfg.device)

    trainer = Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100,
        callbacks=[TQDMProgressBar(refresh_rate=20), LearningRateMonitor(logging_interval='step')],
        default_root_dir='outputs',
        log_every_n_steps=50,
    )

    ckpt_path = f'outputs/lightning_logs/version_{0}/checkpoints/epoch={4}-step={3124}.ckpt'
    # ckpt_path = None
    # trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val, ckpt_path=ckpt_path)
    trainer.validate(model, val_dataloaders=dataloader_val, ckpt_path=ckpt_path)


def data_loading_test(cfg: Config):
    dataset = RandomPoseRegularObjDataset(obj_list=cc.regular_objects, scene_mode=True, device=cfg.device, bg_img_path='/data/coco/train2017')
    # dataset = RandomPoseBOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', scene_mode=False, device=cfg.device, bg_img_path='/data/coco/train2017')
    # dataset = RenderedPoseBOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', scene_mode=True, device=cfg.device)
    # dataset = BOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', device=cfg.device)
    for s in dataset:
        s.visualize()
        a = 0


def numerical_check(cfg, model):
    dataset = BOPObjDataset(obj_list={1: 'ape'}, path='data/BOP/lm', device=cfg.device)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=Sample.collate)
    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_preds.pkl', 'rb') as f:
        gdr_results = pickle.load(f)['ape']
    with open('../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/inference_/lm_13_test/a6-cPnP-lm13_lm_13_test_errors.pkl', 'rb') as f:
        gdr_errors = pickle.load(f)['ape']
    pic_paths = list(gdr_results.keys())
    results = []
    model.eval()
    with torch.no_grad():
        i = 0
        j = 0
        for sample in dataloader:
            pic_path = pic_paths[j]
            id = int(pic_path.split('/')[-1].split('.')[0])
            if i < id:
                i += 1
                continue
            gdr_result = gdr_results[pic_path]
            sample.img = torch.load(f'/data/x/data/{j}.pth').flip(dims=[1])
            # R, t = sample.sanity_check()
            sample = model(sample)
            R, t = sample.pred_cam_R_m2c, sample.pred_cam_t_m2c
            re, te = sample.relative_angle(degree=True), sample.relative_dist(cm=True)
            ad = sample.add_score(dataset.objects_eval)
            proj = sample.proj_dist(dataset.objects_eval)
            gdr_R = torch.tensor(gdr_result['R'], device=cfg.device)[None]
            gdr_t = torch.tensor(gdr_result['t'], device=cfg.device)[None]
            gdr_re = sample.relative_angle(gdr_R, degree=True)
            gdr_te = sample.relative_dist(gdr_t, cm=True)
            gdr_ad = sample.add_score(dataset.objects_eval, gdr_R, gdr_t)
            gdr_proj = sample.proj_dist(dataset.objects_eval, gdr_R, gdr_t)

            result = [[float(re), float(te), float(ad), float(proj)],
                      [float(gdr_re), float(gdr_te), float(gdr_ad), float(gdr_proj)],
                      [gdr_errors['re'][j], gdr_errors['te'][j] * 100, gdr_errors['ad'][j] / dataset.objects_eval[1].diameter, gdr_errors['proj'][j]]]
            result = np.array(result)
            results.append(result)
            i += 1
            j += 1
            # gc.collect()
            # torch.cuda.empty_cache()
    results = np.array(results)
    return results


if __name__ == '__main__':
    main()
