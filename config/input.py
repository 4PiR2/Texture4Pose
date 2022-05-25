import config.const as cc

_base_ = './pipeline.py'

dataloader = dict(
    obj_list=cc.lmo_objects,
    scene_mode=True,
    bg_img_path=None,
)
