_base_ = './const.py'

dataset = dict(
    scene_src=0,  # 0: random, 1: rendered with bop pose, 2: load pics from bop, 3: real exp
    # path='data/BOP/lm',
    path='/data/real_exp/i12P_26mm',
    # obj_list={101: 'sphere', 102: 'cube', 103: 'tetrahedron', 104: 'cylinderstrip', 105: 'sphericon'},
    # obj_list={1: 'ape', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher'},
    obj_list={},
    scene_mode=False,
    bg_img_path='/data/coco/train2017',
    img_render_size=512,
    random_t_depth_range=(.5, 1.2),
    random_t_center_range=(-.7, .7),
    rand_t_inside_cuboid=False,
    vis_ratio_filter_threshold=.5,
    max_dzi_ratio=.25,  # dynamic zoom in
    bbox_zoom_out_ratio=1.5,
    light_max_saturation=1.,
    light_ambient_range=(.5, 1.),
    light_diffuse_range=(0., .3),
    light_specular_range=(0., .2),
    light_shininess_range=(1, 40),
    num_obj=None,
    repeated_sample_obj=True,
    occlusion_size_min=.125,
    occlusion_size_max=.125,
    num_occlusion_per_obj=1,
    min_occlusion_vis_ratio=.5,
    cylinder_strip_thresh_theta=15. * .0174532925199,  # 15 degree
    real_img_ext='heic',
    charuco_w_square=7,
    charuco_h_square=10,
    charuco_square_length=.04,
    cylinder_scale_true=.042,
    cylinder_align_x=3.,
    cylinder_align_y=5.,
    sphericon_scale_true=.05,
    sphericon_align_x=3.,
    sphericon_align_y=5.,
)

augmentation = dict(  # torchvision.transforms.ColorJitter
    brightness=.5,
    contrast=.5,
    saturation=.5,
    hue=.5
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
