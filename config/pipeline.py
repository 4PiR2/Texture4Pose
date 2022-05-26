_base_ = './const.py'

dataset = dict(
    scene_mode=False,
    bg_img_path='/data/coco/train2017',
    img_render_size=512,
    random_t_depth_range=(.5, 1.2),
    vis_ratio_filter_threshold=.5,
    max_dzi_ratio=.25,  # dynamic zoom in
    bbox_zoom_out_ratio=1.5,
    light_ambient_range=(.5, 1.),
    light_diffuse_range=(0., .3),
    light_specular_range=(0., .2),
    light_shininess_range=(40, 80),
    light_color_range=(.9, 1.),
    path=None,
)

augmentation = dict(  # torchvision.transforms.ColorJitter
    brightness=.3,
    contrast=.3,
    saturation=.3,
    hue=0.
)

dataloader = dict(
    batch_size=16,
    train_epoch_len=10000,
    val_epoch_len=100,
)

model = dict(
    gdr_mode=True,
    img_input_size=256,
    pnp_input_size=64,
    pretrain='../GDR-Net/output/gdrn/lm_train_full_wo_region/a6_cPnP_lm13/model_final.pth',

)
