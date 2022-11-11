import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

from dataloader.data_module import LitDataModule
from models.main_model import MainModel
from utils.ckpt_io import CkptIO
from utils.config import Config


def get_cfg(config_path: str = 'config/top.py', args=None) -> Config:
    cfg = Config.fromfile(config_path)
    if args is not None:
        cfg.merge_from_dict(args)
    return cfg


def get_model(cfg: Config = None, ckpt_path: str = None, strict: bool = False):
    if cfg is None:
        cfg = get_cfg('../config/top.py')

    datamodule = LitDataModule(cfg)
    if ckpt_path is not None:
        model = MainModel.load_from_checkpoint(ckpt_path, strict=strict,
                                               cfg=cfg, objects=datamodule.dataset.objects,
                                               objects_eval=datamodule.dataset.objects_eval)
    else:
        model = MainModel(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)
    model = model.to(cfg.device, dtype=cfg.dtype)
    return model, datamodule


def get_trainer(max_epochs: int = 0):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_metric',
        mode='min',
        filename='{epoch:04d}-{val_metric:.4f}',
        save_last=True,
    )

    trainer = Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor(logging_interval='step', log_momentum=False),
            checkpoint_callback,
        ],
        plugins=[CkptIO()],
        default_root_dir='../outputs',
        log_every_n_steps=10,
        num_sanity_val_steps=-1,
    )
    return trainer
