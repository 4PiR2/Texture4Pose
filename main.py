import pickle

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

import config.const as cc
# from dataloader.pose_dataset import BOPObjDataset, RenderedPoseBOPObjDataset
from dataloader.sample import Sample
from models.cdpn.cdpn import CDPN
from utils.ckpt_io import CkptIO
from dataloader.data_module import LitDataModule
# from models.gdr.gdrn import GDRN
from utils.config import Config


def main():
    def setup(args=None) -> Config:
        """Create configs and perform basic setups."""
        cfg = Config.fromfile('config/input.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg = setup()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_metric',
        mode='min',
        filename='{epoch:04d}-{val_metric:.4f}',
        save_last=True,
    )

    # profiler = PyTorchProfiler(filename='profile', emit_nvtx=False)

    trainer = Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor(logging_interval='step', log_momentum=False),
            checkpoint_callback,
        ],
        plugins=[CkptIO()],
        default_root_dir='outputs',
        log_every_n_steps=10,
        # profiler=profiler,
    )

    # ckpt_path = utils.io.find_lightning_ckpt_path('outputs')
    # ckpt_path = 'outputs/lightning_logs/version_14/checkpoints/epoch=0017-val_metric=0.0334.ckpt'
    ckpt_path = 'outputs/lightning_logs/version_24/checkpoints/last.ckpt'
    ckpt_path_n = None

    datamodule = LitDataModule(cfg)

    model = CDPN(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)
    # if cfg.model.pretrain is not None:
    #     model.load_pretrain(cfg.model.pretrain)

    # model = CDPN.load_from_checkpoint(
    #     ckpt_path, cfg=cfg, objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval)

    model = model.to(cfg.device, dtype=cfg.dtype)
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=datamodule)
    # trainer.validate(model, ckpt_path=ckpt_path, datamodule=datamodule)


if __name__ == '__main__':
    main()
