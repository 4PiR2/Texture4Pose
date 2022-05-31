import gc
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

import config.const as cc
from dataloader.pose_dataset import BOPObjDataset, RenderedPoseBOPObjDataset, RandomPoseBOPObjDataset, \
    RandomPoseRegularObjDataset
from dataloader.sample import Sample
from engine.data_module import LitDataModule
from engine.training_module import LitModel
from utils.config import Config
import utils.io


def main():
    def setup(args=None) -> Config:
        """Create configs and perform basic setups."""
        cfg = Config.fromfile('config/input.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg = setup()
    datamodule = LitDataModule(cfg)
    model = LitModel(datamodule.dataset.objects, datamodule.dataset.objects_eval)
    if cfg.model.pretrain is not None:
        model.load_pretrain(cfg.model.pretrain)
    model = model.to(cfg.device)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor='val_metric',
        mode='min',
        filename='{epoch:04d}-{val_metric:.4f}',
        save_last=True,
    )

    from pytorch_lightning.profiler import PyTorchProfiler

    profiler = PyTorchProfiler(filename='profile', emit_nvtx=False)

    trainer = Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            LearningRateMonitor(logging_interval='step', log_momentum=False),
            checkpoint_callback,
        ],
        default_root_dir='outputs',
        log_every_n_steps=50,
        profiler=profiler,
    )

    ckpt_path = utils.io.find_lightning_ckpt_path('outputs')
    # ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=datamodule)
    # trainer.validate(model, ckpt_path=ckpt_path, datamodule=datamodule)


def data_loading_test(cfg):
    dataset = RandomPoseRegularObjDataset(obj_list=cc.regular_objects, scene_mode=True, device=cc.device, bg_img_path='/data/coco/train2017')
    # dataset = RandomPoseBOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', scene_mode=False, device=cc.device, bg_img_path='/data/coco/train2017')
    # dataset = RenderedPoseBOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', scene_mode=True, device=cc.device)
    # dataset = BOPObjDataset(obj_list=cc.lmo_objects, path='data/BOP/lmo', device=cc.device)
    for s in dataset:
        s.visualize()
        a = 0


def numerical_check(cfg, model):
    dataset = BOPObjDataset(obj_list={1: 'ape'}, path='data/BOP/lm', device=cc.device)
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
            gdr_R = torch.tensor(gdr_result['R'], device=cc.device)[None]
            gdr_t = torch.tensor(gdr_result['t'], device=cc.device)[None]
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
