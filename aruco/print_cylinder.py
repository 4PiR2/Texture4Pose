import torch

from dataloader.data_module import LitDataModule
from models.gdr2.gdrn import GDRN
import utils.print_paper
from utils.config import Config


def unroll_cylinder_side(r: float, h: float = None, margin: float = .01, border: int = 1, dpi: int = 72):
    """

    :param r: float, meter
    :param h: float, default = r * 2., meter
    :param margin: float, rate
    :param border: int, px
    :param dpi: int
    :return:
    """

    def xyz_texture():
        return coord_3d / (2. * r) + .5

    def learnt_texture():
        ckpt_path = 'outputs/lightning_logs/version_94/checkpoints/last.ckpt'

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
            # normal,
        ], dim=-3)
        gt_position_info_roi = torch.cat([gt_position_info_roi] \
            + [(x * (torch.pi * 2.)).sin() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32, 64, 128]]] \
            + [(x * (torch.pi * 2.)).cos() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32, 64, 128]]],
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
    img = torch.zeros(3, H + margin, W + border * 2)
    img[:, :H, border: border + W] = learnt_texture()
    img[:, H:] = img[:, H - 1: H]
    img[:, H:, :border] = img[:, H:, border + W:] = 1.
    return img


def get_spectrum_info(y: torch.Tensor, freq_sample: float = None):
    n = y.shape[-1]
    if freq_sample is None:
        freq_sample = float(n)
    y_fft = torch.fft.rfft(y, dim=-1, norm='forward')
    amplitude = y_fft.abs()
    amplitude[..., 1:] *= 2.
    phase = y_fft.angle()
    freq = torch.arange(amplitude.shape[-1], dtype=amplitude.dtype, device=amplitude.device) * (freq_sample / n)
    return freq, amplitude, phase
