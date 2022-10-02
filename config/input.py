_base_ = './pipeline.py'

dataset = dict(
    scene_src=3 * 0,  # 0: random (for training), 3: real exp (for testing)
    # obj_list={101: 'sphere', },
    obj_list={104: 'cylinderstrip', },
    # obj_list={105: 'sphericon', },
    num_obj=16,
    num_occlusion_per_obj=0,
    occlusion_size_max=.5,
    min_occlusion_vis_ratio=.1,
    light_max_saturation=.1,
)

dataloader = dict(
    batch_size=1,
    train_epoch_len=500 * 1,
    val_epoch_len=(200 + dataset['num_obj'] - 1) // dataset['num_obj'],
)

_base_lr = 3e-5

optimizer = dict(
    lr=dict(
        resnet_backbone=_base_lr,
        up_sampling_backbone=_base_lr,
        coord_3d_head=_base_lr,
        texture_net_p=_base_lr / 1.,
        texture_net_v=1e-2,
        secondary_head=_base_lr,
        pnp_net=_base_lr / 1.,
    )
)
