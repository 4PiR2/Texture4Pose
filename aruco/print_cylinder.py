import torch

from dataloader.data_module import LitDataModule
from models.gdr.gdrn import GDRN
import utils.print_paper
from utils.config import Config


def unroll_cylinder_side(r: float, h: float = None, margin: float = .01, dpi: int = 72):
    """

    :param r: float, meter
    :param h: float, default = r * 2., meter
    :param margin: float, rate
    :param dpi: int
    :return:
    """

    def xyz_texture():
        return coord_3d / (2. * r) + .5

    def learnt_texture():
        ckpt_path = 'outputs/lightning_logs/version_89/checkpoints/last.ckpt'

        def setup(args=None) -> Config:
            """Create configs and perform basic setups."""
            cfg = Config.fromfile('config/input.py')
            if args is not None:
                cfg.merge_from_dict(args)
            return cfg

        cfg = setup()
        datamodule = LitDataModule(cfg)
        model = GDRN.load_from_checkpoint(ckpt_path, cfg=cfg, objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval)
        gt_position_info_roi = torch.cat([
            coord_3d / (2. * r) + .5,
            normal,
        ], dim=-3)
        gt_position_info_roi = torch.cat([gt_position_info_roi] \
            + [x.sin() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32]]] \
            + [x.cos() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32]]],
            dim=-3)
        return model.texture_net_p(gt_position_info_roi[None])

    if h is None:
        h = r * 2.
    H = utils.print_paper.meter2px(2. * torch.pi * r, dpi)
    W = utils.print_paper.meter2px(h, dpi)
    margin = int(torch.ceil(torch.tensor(H * margin)))
    coord_3d = torch.empty(3, H, W)
    normal = torch.empty_like(coord_3d)
    theta = torch.linspace(0., 2. * torch.pi, H)[..., None]
    normal[0] = theta.cos()
    normal[1] = theta.sin()
    normal[2] = 0.
    coord_3d[:2] = normal[:2] * r
    coord_3d[2] = torch.linspace(-.5 * h, .5 * h, W)
    img = torch.zeros(3, H + margin, W + 2)
    img[:, :-margin, 1:-1] = learnt_texture()
    img[:, -margin:] = img[:, -margin-1:-margin]
    img[:, -margin:, 0] = img[:, -margin:, -1] = 1.
    return img
