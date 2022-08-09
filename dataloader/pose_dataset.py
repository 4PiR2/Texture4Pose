from typing import Iterator

import torch.utils.data

import config.const as cc
import dataloader.datapipe.functional_bop
import dataloader.datapipe.functional_custom
from dataloader.datapipe.helper import SampleSource
from dataloader.sample import Sample


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.IterableDataset, len: int):
        self.dataset_iter: Iterator = iter(dataset)
        self.len: int = len

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int = None) -> Sample:
        return self.dataset_iter.__next__()


def random_scene_any_obj_dp(
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True,
    bg_img_path=None, img_render_size=512, crop_out_size=256, cam_K=cc.lm_cam_K,
    random_t_depth_range=(.5, 1.2), vis_ratio_filter_threshold=.5, max_dzi_ratio=.25,
    bbox_zoom_out_ratio=1.5, light_max_saturation=1., light_ambient_range=(.5, 1.),
    light_diffuse_range=(0., .3), light_specular_range=(0., .2), light_shininess_range=(40, 80),
    num_obj=None, repeated_sample_obj=False, occlusion_size_min=.125, occlusion_size_max=.5,
    num_occlusion_per_obj=1, min_occlusion_vis_ratio=.5, **kwargs,
):
    dp = SampleSource(dtype=dtype, device=device, scene_mode=scene_mode, img_render_size=img_render_size)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=path)
    dp = dp.set_mesh_info()
    dp = dp.set_static_camera(cam_K=cam_K)
    dp = dp.rand_select_objs(num_obj=num_obj, repeated_sample_obj=repeated_sample_obj)
    dp = dp.rand_gt_translation(random_t_depth_range=random_t_depth_range, cuboid=False)
    # dp = dp.rand_gt_rotation()
    dp = dp.rand_gt_rotation_cylinder(thresh_theta=15. * torch.pi / 180.)
    dp = dp.render_scene()
    dp = dp.gen_mask()
    dp = dp.compute_vis_ratio()
    dp = dp.filter_vis_ratio(vis_ratio_filter_threshold=vis_ratio_filter_threshold)
    dp = dp.gen_bbox()
    dp = dp.dzi_bbox(max_dzi_ratio=max_dzi_ratio, bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.crop_roi_basic(out_size=crop_out_size)
    dp = dp.rand_occlude(occlusion_size_min=occlusion_size_min, occlusion_size_max=occlusion_size_max,
                         num_occlusion_per_obj=num_occlusion_per_obj, min_occlusion_vis_ratio=min_occlusion_vis_ratio,
                         batch_occlusion=1)
    dp = dp.rand_lights(light_max_saturation=light_max_saturation, light_ambient_range=light_ambient_range,
                        light_diffuse_range=light_diffuse_range,
                        light_specular_range=light_specular_range, light_shininess_range=light_shininess_range)
    dp = dp.apply_lighting(batch_lighting=1)
    dp = dp.rand_bg(bg_img_path=bg_img_path)
    dp = dp.crop_roi_bg(out_size=crop_out_size)
    dp = dp.apply_bg()  # convert mask_vis back to bool
    dp = dp.gen_coord_2d(width=img_render_size, height=img_render_size)
    dp = dp.crop_coord_2d(out_size=crop_out_size)
    dp = dp.render_img()
    # dp = dp.augment_img(transform=transform)
    return dp


def rendered_scene_bop_obj_dp(
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True,
    crop_out_size=64,
    vis_ratio_filter_threshold=.5, max_dzi_ratio=.25,
    bbox_zoom_out_ratio=1.5, light_max_saturation=1., light_ambient_range=(.5, 1.),
    light_diffuse_range=(0., .3), light_specular_range=(0., .2), light_shininess_range=(40, 80), **kwargs,
):
    dp = SampleSource(dtype=dtype, device=device, scene_mode=scene_mode, img_render_size=640)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=path)
    dp = dp.load_bop_scene()
    # dp = dp.rand_scene_id()
    dp = dp.set_pose()
    dp = dp.set_camera()
    dp = dp.set_bg()
    dp = dp.set_mesh_info()
    dp = dp.render_scene()
    dp = dp.gen_mask()
    dp = dp.set_mask(bitwise_and_with_existing=True)
    dp = dp.set_bbox()
    dp = dp.remove_item_id()
    dp = dp.filter_vis_ratio(vis_ratio_filter_threshold=vis_ratio_filter_threshold)
    dp = dp.dzi_bbox(max_dzi_ratio=0., bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.crop_roi_basic(out_size=crop_out_size)
    dp = dp.rand_lights(light_max_saturation=0., light_ambient_range=(1., 1.),
                        light_diffuse_range=(0., 0.),
                        light_specular_range=(0., 0.), light_shininess_range=(40, 40))
    dp = dp.apply_lighting(batch_lighting=1)
    dp = dp.crop_roi_bg(out_size=crop_out_size)
    dp = dp.apply_bg()  # convert mask_vis back to bool
    dp = dp.gen_coord_2d(width=640, height=480)
    dp = dp.crop_coord_2d(out_size=crop_out_size)
    dp = dp.render_img()
    # dp = dp.augment_img(transform=transform)
    return dp


def bop_scene_bop_obj_dp(
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True,
    crop_out_size=64,
    vis_ratio_filter_threshold=.5, max_dzi_ratio=.25,
    bbox_zoom_out_ratio=1.5, **kwargs,
):
    dp = SampleSource(dtype=dtype, device=device, scene_mode=scene_mode, img_render_size=640)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=path)
    dp = dp.load_bop_scene()
    dp = dp.rand_scene_id()
    dp = dp.set_pose()
    dp = dp.set_camera()
    dp = dp.set_bg()
    dp = dp.set_mesh_info()
    dp = dp.set_depth()
    dp = dp.set_mask()
    dp = dp.set_bbox()
    dp = dp.remove_item_id()
    dp = dp.filter_vis_ratio(vis_ratio_filter_threshold=vis_ratio_filter_threshold)
    dp = dp.dzi_bbox(max_dzi_ratio=0., bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.gen_coord_2d(width=640, height=480)
    dp = dp.crop_roi_bop(out_size=crop_out_size)
    dp = dp.set_coord_3d()
    # dp = dp.augment_img(transform=transform)
    return dp
