import matplotlib.pyplot as plt
import torch
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

from dataloader.data_module import LitDataModule
from models.main_model import MainModel
import realworld.charuco_board
import realworld.print_unroll
from utils.ckpt_io import CkptIO
from utils.config import Config
import utils.print_paper


def print_board():
    realworld.charuco_board.ChArUcoBoard(7, 10, .04).to_paper_pdf('/home/user/Desktop/1.pdf', paper_size='a3')


def print_cylinder_strip(model=None):
    dpi = 300
    img_0 = realworld.print_unroll.unroll_cylinder_strip(scale=.0402, dpi=dpi, model=None)
    img_1 = realworld.print_unroll.unroll_cylinder_strip(scale=.0402, dpi=dpi, model=model)
    img_2 = realworld.print_unroll.unroll_cylinder_strip(scale=.0402, dpi=dpi, model=None, margin=0.)
    cb_num_cycles = 2
    img_2 = (img_2 * (cb_num_cycles * 2)).int() % 2
    img_2bg = torch.ones_like(img_0)
    img_2bg[..., :img_2.shape[-2], :img_2.shape[-1]] = img_2
    margin = .05
    white_space = torch.ones_like(img_0)[..., :int(img_0.shape[-1] * margin) + 1]
    img_3 = img_0
    img = torch.cat([img_0, white_space, img_1, white_space, img_2bg, white_space, img_3], dim=-1)
    img = img.transpose(-2, -1).flip(dims=[-1])
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/c2.pdf', dpi=dpi, paper_size='a3')


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
    utils.print_paper.print_tensor_to_paper_pdf(img_100, '/home/user/Desktop/s1.pdf', dpi=dpi)


def print_sphericon_a3(model=None):
    dpi = 300
    img_0 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model=None)
    img_1 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model=model)
    img_2 = realworld.print_unroll.unroll_sphericon(scale=.05, theta=2.25, dpi=dpi, model='cb')
    margin = .01
    white_space = torch.ones_like(img_0)[..., :int(img_0.shape[-1] * margin) + 1]
    img = torch.cat([img_1, white_space, img_1], dim=-1)
    img = img.transpose(-2, -1).flip(dims=[-1])
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/s2.pdf', dpi=dpi, paper_size='a3')


def main():
    # ckpt_path = utils.io.find_lightning_ckpt_path('outputs')

    # 104 xyz
    # ckpt_path = 'outputs/lightning_logs/version_279/checkpoints/epoch=0127-val_metric=1.9202.ckpt'
    # 104 siren
    # ckpt_path = 'outputs/lightning_logs/version_268/checkpoints/epoch=0043-val_metric=0.3965.ckpt'
    # 104 cb
    # ckpt_path = 'outputs/lightning_logs/version_273/checkpoints/epoch=0122-val_metric=0.4129.ckpt'

    # 105 xyz
    # ckpt_path = 'outputs/lightning_logs/version_291/checkpoints/epoch=0119-val_metric=1.5092.ckpt'
    # 105 siren
    # ckpt_path = 'outputs/lightning_logs/version_301/checkpoints/epoch=0007-val_metric=0.4302.ckpt'
    # 105 cb
    # ckpt_path = 'outputs/lightning_logs/version_282/checkpoints/epoch=0154-val_metric=0.3779.ckpt'

    # 101 xyz
    ckpt_path = 'outputs/lightning_logs/version_300/checkpoints/epoch=0114-val_metric=2.8337.ckpt'
    # 101 siren
    # ckpt_path = 'outputs/lightning_logs/version_327/checkpoints/epoch=0164-val_metric=0.3148.ckpt'
    # 101 cb
    # ckpt_path = 'outputs/lightning_logs/version_328/checkpoints/epoch=0120-val_metric=0.2599.ckpt'
    # ckpt_path = None

    only_load_weights = True
    max_epochs = 200
    do_fit = True
    do_val = not do_fit

    def setup(args=None) -> Config:
        cfg = Config.fromfile('config/top.py')
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
        default_root_dir='outputs',
        log_every_n_steps=10,
        num_sanity_val_steps=-1,
    )

    datamodule = LitDataModule(cfg)

    if only_load_weights and ckpt_path is not None:
        model = MainModel.load_from_checkpoint(ckpt_path, strict=False,
            cfg=cfg, objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval)
    else:
        model = MainModel(cfg, datamodule.dataset.objects, datamodule.dataset.objects_eval)


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

    # print_cylinder_strip(model)
    # print_sphericon_a3(model)

    model = model.to(cfg.device, dtype=cfg.dtype)

    # from pytorch3d.structures import Meshes
    # from pytorch3d.renderer import TexturesVertex
    # from pytorch3d.vis.plotly_vis import plot_scene
    #
    # mesh: Meshes = datamodule.dataset.objects[105].mesh
    # verts = mesh.verts_packed()[..., None, None] * 10. + .5
    # normals = mesh.verts_normals_packed()[..., None, None]
    # mask = torch.ones(len(verts), 1, 1, 1, dtype=torch.bool, device=verts.device)
    # model.eval()
    # with torch.no_grad():
    #     texture = model.forward_texture(coord_3d_normalized=verts, normal=normals, mask=mask)
    # mesh.textures = TexturesVertex(texture[None, ..., 0, 0])
    # plot_scene({'subplot1': {'m': mesh}}).show()

    if do_fit:
        trainer.fit(model, ckpt_path=None if only_load_weights else ckpt_path, datamodule=datamodule)
    if do_val:
        trainer.validate(model, ckpt_path=None if only_load_weights else ckpt_path, datamodule=datamodule)

    exit(1)

    from dataloader.pose_dataset import real_scene_regular_obj_dp
    dp = real_scene_regular_obj_dp(path='/data/real_exp/i12P_26mm', obj_list={105: 'sphericon'},
                                   texture='siren', num_pose_augmentation=0)
    model.eval()
    with torch.no_grad():
        i = 0
        for y in tqdm.tqdm(dp):
            y = model(y)
            fig = y.visualize(return_figs=True)[0]
            fig.savefig(f'/home/user/Desktop/tmp/f{i}.png')
            # fig.show()
            i += 1


if __name__ == '__main__':
    main()
