_base_ = './const.py'

dataloader = dict(
    # dynamic zoom in
    max_dzi_ratio=.25,

    bbox_zoom_out=1.5,
    vis_ratio_threshold=.5,
    t_depth_min=.5,
    t_depth_max=1.2,
)

model = dict(
    gdr_mode=True,

    img_render_size=512,
    img_input_size=256,
    pnp_input_size=64,
)
