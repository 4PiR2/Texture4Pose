_base_ = './const.py'

dataset = dict(
    scene_mode=False,
    bg_img_path='/data/coco/train2017',
    img_render_size=512,
    random_t_depth_range=(.5, 1.2),
    vis_ratio_filter_threshold=.5,
    max_dzi_ratio=.25,  # dynamic zoom in
    bbox_zoom_out_ratio=1.5,
    light_max_saturation=1.,
    light_ambient_range=(.5, 1.),
    light_diffuse_range=(0., .3),
    light_specular_range=(0., .2),
    light_shininess_range=(40, 80),
    num_obj=None,
    repeated_sample_obj=False,
    path='data/BOP/lm',
    occlusion_size_min=.125,
    occlusion_size_max=.5,
    num_occlusion_per_obj=4,
    min_occlusion_vis_ratio=.25,
    bop_scene=False,
)

augmentation = dict(  # torchvision.transforms.ColorJitter
    brightness=.3,
    contrast=.3,
    saturation=.3,
    hue=0.
)

dataloader = dict(
    batch_size=2,
    train_epoch_len=10000,
    val_epoch_len=100,
)

model = dict(
    gdr_mode=True,
    img_input_size=256,
    pnp_input_size=64,
    pretrain='../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth',
)
