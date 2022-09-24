import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

import config.const as cc
from dataloader.data_module import LitDataModule
from dataloader.sample import Sample
from models.cdpn.cdpn import CDPN
from models.cdpn.cdpn2 import CDPN2
from models.drn.drn import DRN
# from models.gdr.gdrn import GDRN
from models.gdr2.gdrn import GDRN
from models.surfemb.surfemb import SurfEmb
import realworld.charuco_board
import realworld.print_unroll
from utils.ckpt_io import CkptIO
from utils.config import Config
import utils.print_paper


def print_board():
    realworld.charuco_board.ChArUcoBoard(7, 10, .04).to_paper_pdf('/home/user/Desktop/1.pdf', paper_size='a3')


def print_cylinder_strip():
    dpi = 300
    img_84 = realworld.print_unroll.unroll_cylinder_strip(scale=.042, dpi=dpi)
    img_82 = realworld.print_unroll.unroll_cylinder_strip(scale=.041, dpi=dpi)
    img = utils.print_paper.make_grid(img_84, (1, 2), margin=.05)
    img[..., :img_84.shape[-1]] = 1.
    img[..., :img_82.shape[-2], :img_82.shape[-1]] = img_82
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/4.pdf', dpi=dpi)


def spectrum_analysis():
    img_100 = realworld.print_unroll.unroll_cylinder_strip(scale=.05, margin=0., border=0, dpi=300)
    # img_100 = img_100.transpose(-2, -1).flip(-2)
    from utils.image_2d import visualize
    visualize(img_100)
    freq, amplitude, phase = realworld.print_unroll.get_spectrum_info(img_100.detach())
    cutoff = 500
    for channel in range(3):
        plt.plot(freq[:cutoff], amplitude[channel, :, :cutoff].mean(dim=-2), c=['r', 'g', 'b'][channel])
        plt.show()
    plt.plot(freq[:cutoff], amplitude[:, :, :cutoff].mean(dim=[-3, -2]), c='k')
    plt.show()


def print_sphericon():
    dpi = 300
    img = realworld.print_unroll.unroll_sphericon(scale=.05, dpi=dpi)
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/s1.pdf', dpi=dpi)


def main():
    print_sphericon()
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
        max_epochs=110,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor(logging_interval='step', log_momentum=False),
            checkpoint_callback,
        ],
        plugins=[CkptIO()],
        default_root_dir='outputs',
        log_every_n_steps=10,
        # profiler=profiler,
        # gradient_clip_val=0.,
        # gradient_clip_algorithm='value',
        # stochastic_weight_avg=True,
        # terminate_on_nan=True,
    )

    # ckpt_path = utils.io.find_lightning_ckpt_path('outputs')
    # ckpt_path = 'outputs/lightning_logs/version_14/checkpoints/epoch=0017-val_metric=0.0334.ckpt'
    ckpt_path = 'outputs/lightning_logs/version_138/checkpoints/last.ckpt'
    ckpt_path_n = None

    datamodule = LitDataModule(cfg)

    model = GDRN(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)

    # state_dict = torch.load('outputs/lightning_logs/version_34/checkpoints/epoch=0037-val_metric=0.0432.ckpt')['state_dict']
    # state_dict2 = {}
    # for k, v in state_dict.items():
    #     if k.startswith('rotation_backbone'):
    #         state_dict2['guide_'+k] = v
    #     else:
    #         state_dict2[k] = v
    # model.load_state_dict(state_dict2, strict=False)

    # if cfg.model.pretrain is not None:
    #     model.load_pretrain(cfg.model.pretrain)

    # model = GDRN.load_from_checkpoint(ckpt_path, strict=True,
    #     cfg=cfg, objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval)

    model = model.to(cfg.device, dtype=cfg.dtype)
    trainer.fit(model, ckpt_path=ckpt_path_n, datamodule=datamodule)
    # trainer.validate(model, ckpt_path=ckpt_path, datamodule=datamodule)

    exit(1)

    from dataloader.pose_dataset import real_scene_regular_obj_dp
    from utils.image_2d import visualize
    dp = real_scene_regular_obj_dp(path='/data/real_exp/i12P_26mm', obj_list={104: 'cylinderstrip'},)
    # with open('/home/user/Desktop/x.pkl', 'rb') as f:
    #     x = pickle.load(f)
    model.eval()
    with torch.no_grad():
        i = 0
        for y in dp:
            y = model(y)
            fig = y.visualize(return_figs=True)[0]
            fig.savefig(f'/home/user/Desktop/f{i}.png')
            fig.show()
            i += 1


if __name__ == '__main__':
    main()
