import glob
import os
import re

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import tqdm

from dataloader.data_module import LitDataModule
from models.main_model import MainModel
import realworld.print_unroll
from utils.config import Config
from realworld.proj_sphere import equirectangular


def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1, sizes[0] / sizes[1], forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, transparent=True, dpi=sizes[1])
    # plt.show()
    plt.close()


def visualize_texture(versions: list = [310, 325]):
    ckpt_path_list = []
    for v in versions:
        p = os.path.join('outputs/lightning_logs/version_{}'.format(v), 'checkpoints_texture_net_p')
        ckpt_path_list.extend([os.path.join(p, pp) for pp in os.listdir(p)])

    pattern = re.compile(r'/version_(\d+)/checkpoints.*/epoch=(\d+)-step=(\d+)\.ckpt$')
    paths = []
    for p in ckpt_path_list:
        m = pattern.search(p)
        v, e, s = m.group(1), m.group(2), m.group(3)
        v, e, s = int(v), int(e), int(s)
        paths.append([v, e, s, p])
    paths.sort()

    ckpt_path_list = []
    last_v = versions[0]
    for v, e, s, p in paths:
        if v == last_v:
            ckpt_path_list.append(p)
        last_v = v


    def setup(args=None) -> Config:
        cfg = Config.fromfile('config/top.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg = setup()
    datamodule = LitDataModule(cfg)
    cfg.model.texture_mode = 'siren'
    cfg.model.pnp_mode = None
    model = MainModel(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)

    i = 0
    for p in tqdm.tqdm(ckpt_path_list):
        model.texture_net_p = torch.load(p).cpu()
        # img = realworld.print_unroll.unroll_sphericon(scale=.05, theta=0., dpi=72, model=model)[0]
        # img = realworld.print_unroll.unroll_cylinder_strip(scale=.05, margin=0., border=0, dpi=72, model=model)
        # img = img.transpose(-2, -1).flip(dims=[-2])
        img = equirectangular(2000)
        img = model.texture_net_p.forward(torch.cat([img, img], dim=0))
        if i == 0:
            *_, H, W = img.shape
            out = cv2.VideoWriter('/home/user/Desktop/tmp/output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (W, H))
        img = (img * 255.).round().to(torch.uint8).permute(1, 2, 0).numpy()
        out.write(img[..., ::-1])
        save_image(img, os.path.join('/home/user/Desktop/tmp', f'{i:03}.png'))
        i += 1
    out.release()
    a = 0


if __name__ == '__main__':
    # fl = [i for i in glob.glob('/home/user/Desktop/tmp/*.png')]
    # fl.sort()
    # # frameSize = (896, 312)
    # frameSize = (891, 283)
    # out = cv2.VideoWriter('/home/user/Desktop/tmp/output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, frameSize)
    # for filename in fl:
    #     print(filename)
    #     img = cv2.imread(filename)
    #     out.write(img)
    # out.release()

    visualize_texture()
