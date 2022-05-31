# import config.const as cc

_base_ = './pipeline.py'

dataset = dict(
    obj_list={101: 'sphere'},
    # cam_K=cc.lm_cam_K,
    scene_mode=False,
)
