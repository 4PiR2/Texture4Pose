# import config.const as cc

_base_ = './pipeline.py'

dataset = dict(
    # obj_list={101: 'sphere', 102: 'cube', 103: 'tetrahedron'},
    # obj_list={1: 'ape', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher'},
    obj_list={104: 'cylinderside',},
    # cam_K=cc.lm_cam_K,
    scene_mode=False,
    num_obj=2,
    repeated_sample_obj=True,
    occlusion_size_max=.5,
    # path='/data/real_exp/i12P_26mm',
    path='data/BOP/lm',
    bop_scene=0,  # 0: random, 1: rendered, 2: load from bop, 3: real
)

dataloader = dict(
    batch_size=8,
    train_epoch_len=500 * 8,
    val_epoch_len=200 // dataset['num_obj'],
)
