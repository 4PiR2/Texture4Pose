_base_ = './default.py'

dataset = dict(
    # scene_src=3 * 0,  # 0: random (for training), 3: real exp (for testing)
    scene_src=4,
    # obj_list={101: 'sphere', },
    # obj_list={104: 'cylinderstrip', },
    obj_list={105: 'sphericon', },
    # num_obj=16,
    num_obj=16,
    num_pose_augmentation=8,
    occlusion_probability_eval=0.,
    max_dzi_ratio_eval=.25,
    random_t_depth_range=(.1, 1.2),
)

dataloader = dict(
    batch_size=2,
    train_epoch_len=500 * 1,
    val_epoch_len=(200 + dataset['num_obj'] - 1) // dataset['num_obj'],
)

model = dict(
    # texture_mode='xyz',
    texture_mode='siren',
    # texture_mode='cb',
    # texture_mode='scb',
    # freeze_texture_net_p=False,
    freeze_texture_net_p=True,
    # pnp_mode=None,
    eval_augmentation=True and dataset['scene_src'] != 3,
    texture=dict(
        siren_first_omega_0=1.,
        cb_num_cycles=2,
    ),
    coord_3d_loss_weights=[1.],
    coord_3d_loss_weight_step=1,
    pnp=dict(
        epro_loss_weights=[.02, 1.],
        # epro_loss_weights=[1.],
    )
)

_base_lr = 3e-5

optimizer = dict(
    lr=dict(
        resnet_backbone=_base_lr,
        up_sampling_backbone=_base_lr,
        coord_3d_head=_base_lr,
        texture_net_p=_base_lr / 10.,
        texture_net_v=1e-2,
        secondary_head=_base_lr,
        pnp_net=_base_lr / 10.,
    )
)

augmentation = dict(
    color_jitter=dict(
        # hue=0.,
    ),
)
