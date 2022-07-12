# import config.const as cc

_base_ = './pipeline.py'

dataset = dict(
    # obj_list={101: 'sphere', 102: 'cube', 103: 'tetrahedron'},
    obj_list={1: 'ape'},
    # cam_K=cc.lm_cam_K,
    scene_mode=False,
    num_obj=2,
    repeated_sample_obj=True,
    occlusion_size_max=.5,
    path='data/BOP/lm',
    bop_scene=True,
)

dataloader = dict(
    batch_size=8,
    train_epoch_len=500 * 8,
    val_epoch_len=200 // dataset['num_obj'],
)
