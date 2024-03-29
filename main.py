import os.path

import matplotlib.pyplot as plt
import torch

import realworld.charuco_board
import realworld.print_unroll
import utils.get_model
import utils.print_paper


def print_board():
    realworld.charuco_board.ChArUcoBoard(7, 10, .04).to_paper_pdf('/home/user/Desktop/1.pdf', paper_size='a3')


def print_cylinder_strip(model):
    dpi = 300
    img_100 = realworld.print_unroll.unroll_cylinder_strip(scale=.0395, dpi=dpi, model=model)
    img_101 = realworld.print_unroll.unroll_cylinder_strip(scale=.04, dpi=dpi, model=model)
    img_100bg = torch.ones_like(img_101)
    img_100bg[..., :img_100.shape[-2], :img_100.shape[-1]] = img_100
    margin = .05
    white_space = torch.ones_like(img_101)[..., :int(img_101.shape[-1] * margin) + 1]
    img = torch.cat([img_100bg, white_space, img_101], dim=-1)
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/c1.pdf', dpi=dpi)


def print_cylinder_strip_a3(model=None):
    dpi = 300
    img_0 = realworld.print_unroll.unroll_cylinder_strip(scale=.04, dpi=dpi, model=None)
    img_1 = realworld.print_unroll.unroll_cylinder_strip(scale=.04, dpi=dpi, model=model)
    img_2 = realworld.print_unroll.unroll_cylinder_strip(scale=.04, dpi=dpi, model=None, margin=0.)
    cb_num_cycles = 2
    img_2 = (img_2 * (cb_num_cycles * 2)).int() % 2
    img_2bg = torch.ones_like(img_0)
    img_2bg[..., :img_2.shape[-2], :img_2.shape[-1]] = img_2
    margin = .05
    white_space = torch.ones_like(img_0)[..., :int(img_0.shape[-1] * margin) + 1]
    img_3 = img_0
    img = torch.cat([img_0, white_space, img_1, white_space, img_2bg, white_space, img_3], dim=-1)
    img = img.transpose(-2, -1).flip(dims=[-1])
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/c1.pdf', dpi=dpi, paper_size='a3')


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
    img_100 = realworld.print_unroll.unroll_sphericon(scale=.05, dpi=dpi)
    utils.print_paper.print_tensor_to_paper_pdf(img_100, '/home/user/Desktop/s.pdf', dpi=dpi)


def print_sphericon_a3(model=None):
    dpi = 300
    img_0 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model=None)
    img_1 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model=model)
    img_2 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model='cb')
    margin = .01
    white_space = torch.ones_like(img_0)[..., :int(img_0.shape[-1] * margin) + 1]
    img = torch.cat([img_0, white_space, img_0], dim=-1)
    img = img.transpose(-2, -1).flip(dims=[-1])
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/s1.pdf', dpi=dpi, paper_size='a3')
    img = torch.cat([img_1, white_space, img_2], dim=-1)
    img = img.transpose(-2, -1).flip(dims=[-1])
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/s2.pdf', dpi=dpi, paper_size='a3')


def main(obj: int, texture: str, do_fit: bool = False, do_val_synt: bool = False, do_val_real: bool = False,
         do_print: bool = False, data_path: str = None, bg_img_path: str = None):

    only_load_weights = True
    max_epochs = 200

    cfg = utils.get_model.get_cfg('config/top.py')

    if obj == 101:
        cfg.dataset.obj_list = {101: 'sphere', }
        cfg.augmentation.color_jitter.hue = 0.
    if obj == 104:
        cfg.dataset.obj_list = {104: 'cylinderstrip', }
    if obj == 105:
        cfg.dataset.obj_list = {105: 'sphericon', }

    cfg.model.texture_mode = texture

    if do_val_synt:
        cfg.dataloader.val_epoch_len = (200 * 8 + cfg.dataset.num_obj - 1) // cfg.dataset.num_obj

    if do_val_real:
        cfg.dataset.scene_src = 4
        cfg.model.eval_augmentation = False

    cfg.dataset.path = data_path
    cfg.dataset.bg_img_path = bg_img_path

    ckpt_path = os.path.join('weights', f'{obj}{texture}.ckpt')

    model, datamodule = utils.get_model.get_model(cfg, ckpt_path=ckpt_path, strict=False)

    if do_print:
        if obj == 104:
            print_cylinder_strip(model.to('cpu'))
        if obj == 105:
            print_sphericon_a3(model.to('cpu'))

    if do_fit or do_val_synt or do_val_real:
        trainer = utils.get_model.get_trainer(max_epochs)
        if do_fit:
            trainer.fit(model, ckpt_path=None if only_load_weights else ckpt_path, datamodule=datamodule)
        if do_val_synt or do_val_real:
            trainer.validate(model, ckpt_path=None if only_load_weights else ckpt_path, datamodule=datamodule)


if __name__ == '__main__':
    main(104, 'sa', do_fit=False, do_val_synt=False, do_val_real=False, do_print=False,
         data_path='/data/real_exp/i12P_26mm', bg_img_path='/data/coco/train2017')
