import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor

from dataloader.data_module import LitDataModule
from models.main_model import MainModel
from utils.config import Config


def omega_scan():
    def setup(args=None) -> Config:
        cfg = Config.fromfile('config/top.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg = setup()

    datamodule = LitDataModule(cfg)

    cfg.model.texture_mode = 'siren'
    cfg.model.pnp_mode = None
    for omega in [2 ** i for i in [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]]:
        trainer = Trainer(
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=10,
            callbacks=[
                TQDMProgressBar(refresh_rate=1),
                LearningRateMonitor(logging_interval='step', log_momentum=False),
            ],
            default_root_dir='outputs',
            log_every_n_steps=10,
            num_sanity_val_steps=-1,
        )
        cfg.model.texture.siren_first_omega_0 = omega
        model = MainModel(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)
        model = model.to(cfg.device, dtype=cfg.dtype)
        trainer.fit(model, ckpt_path=None, datamodule=datamodule)


if __name__ == '__main__':
    omega_scan()
