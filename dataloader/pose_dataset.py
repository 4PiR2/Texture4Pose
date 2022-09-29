from typing import Iterator

import torch.utils.data

import config.const as cc
import dataloader.datapipe.functional_bop
import dataloader.datapipe.functional_real
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


def random_scene_any_obj_dp0(  # scene_src == 0: random (cropping based, obsoleted)
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True, bg_img_path=None, img_render_size=512,
    crop_out_size=256, cam_K=cc.lm_cam_K, random_t_depth_range=(.5, 1.2), rand_t_inside_cuboid=False,
    vis_ratio_filter_threshold=.5, max_dzi_ratio=.25, bbox_zoom_out_ratio=1.5, light_max_saturation=1.,
    light_ambient_range=(.5, 1.), light_diffuse_range=(0., .3), light_specular_range=(0., .2),
    light_shininess_range=(40, 80), num_obj=None, repeated_sample_obj=False, occlusion_size_min=.125,
    occlusion_size_max=.5, num_occlusion_per_obj=1, min_occlusion_vis_ratio=.5,
    cylinder_strip_thresh_theta=15. * torch.pi / 180., **kwargs,
):
    dp = SampleSource(dtype=dtype, device=device, scene_mode=scene_mode, img_render_size=img_render_size)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=path)
    dp = dp.set_mesh_info()
    dp = dp.set_static_camera(cam_K=cam_K, orig=False)
    dp = dp.rand_select_objs(num_obj=num_obj, repeated_sample_obj=repeated_sample_obj)
    dp = dp.rand_gt_translation_inside_camera(random_t_depth_range=random_t_depth_range, cuboid=rand_t_inside_cuboid)
    dp = dp.rand_gt_rotation_real(cylinder_strip_thresh_theta=cylinder_strip_thresh_theta)
    dp = dp.render_scene()
    dp = dp.gen_mask()
    dp = dp.compute_vis_ratio()
    dp = dp.filter_vis_ratio(vis_ratio_filter_threshold=vis_ratio_filter_threshold)
    dp = dp.gen_bbox()
    dp = dp.dzi_bbox(max_dzi_ratio=max_dzi_ratio, bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.crop_roi_basic(out_size=crop_out_size, delete_original=True)
    dp = dp.rand_occlude(occlusion_size_min=occlusion_size_min, occlusion_size_max=occlusion_size_max,
                         num_occlusion_per_obj=num_occlusion_per_obj, min_occlusion_vis_ratio=min_occlusion_vis_ratio,
                         batch_occlusion=1)
    dp = dp.rand_lights(light_max_saturation=light_max_saturation, light_ambient_range=light_ambient_range,
                        light_diffuse_range=light_diffuse_range, light_specular_range=light_specular_range,
                        light_shininess_range=light_shininess_range)
    dp = dp.apply_lighting(batch_lighting=1)
    dp = dp.rand_bg(bg_img_path=bg_img_path)
    dp = dp.crop_roi_bg(out_size=crop_out_size, delete_original=True)
    dp = dp.apply_bg()  # convert mask_vis back to bool
    dp = dp.gen_coord_2d(width=img_render_size, height=img_render_size)
    dp = dp.crop_coord_2d(out_size=crop_out_size)
    dp = dp.render_img()
    return dp


def random_scene_any_obj_dp(  # scene_src == 0: random (adaptive camera intrinsics)
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, bg_img_path=None, crop_out_size=256,
    random_t_depth_range=(.5, 1.2), random_t_center_range=(-.7, .7), rand_t_inside_cuboid=False, max_dzi_ratio=.25,
    bbox_zoom_out_ratio=1.5, light_max_saturation=1., light_ambient_range=(.5, 1.), light_diffuse_range=(0., .3),
    light_specular_range=(0., .2), light_shininess_range=(40, 80), num_obj=None, repeated_sample_obj=False,
    occlusion_size_min=.125, occlusion_size_max=.5, num_occlusion_per_obj=1, min_occlusion_vis_ratio=.5,
    cylinder_strip_thresh_theta=15. * torch.pi / 180., **kwargs,
):
    dp = SampleSource(dtype=dtype, device=device, scene_mode=False, img_render_size=crop_out_size)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=path)
    dp = dp.set_mesh_info()
    dp = dp.rand_select_objs(num_obj=num_obj, repeated_sample_obj=repeated_sample_obj)
    dp = dp.rand_gt_translation(random_t_depth_range=random_t_depth_range, random_t_center_range=random_t_center_range,
                                cuboid=rand_t_inside_cuboid)
    dp = dp.rand_gt_rotation_real(cylinder_strip_thresh_theta=cylinder_strip_thresh_theta)
    dp = dp.gen_bbox_proj()
    dp = dp.dzi_bbox(max_dzi_ratio=max_dzi_ratio, bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.set_roi_camera()
    dp = dp.render_scene()
    dp = dp.gen_mask()
    dp = dp.rand_bg(bg_img_path=bg_img_path)
    dp = dp.crop_roi_dummy(delete_original=True)
    dp = dp.normalize_normal()
    dp = dp.rand_occlude(occlusion_size_min=occlusion_size_min, occlusion_size_max=occlusion_size_max,
                         num_occlusion_per_obj=num_occlusion_per_obj, min_occlusion_vis_ratio=min_occlusion_vis_ratio,
                         batch_occlusion=1)
    dp = dp.rand_lights(light_max_saturation=light_max_saturation, light_ambient_range=light_ambient_range,
                        light_diffuse_range=light_diffuse_range, light_specular_range=light_specular_range,
                        light_shininess_range=light_shininess_range)
    dp = dp.apply_lighting(batch_lighting=1)
    dp = dp.apply_bg()
    dp = dp.gen_coord_2d_bbox()
    dp = dp.set_static_camera(cam_K=torch.eye(3), orig=True)
    dp = dp.calibrate_bbox()
    dp = dp.render_img()
    return dp


def rendered_scene_bop_obj_dp(  # scene_src == 1: rendered with bop pose (cropping based)
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True, crop_out_size=64,
    vis_ratio_filter_threshold=.5, bbox_zoom_out_ratio=1.5, **kwargs,
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
    dp = dp.crop_roi_basic(out_size=crop_out_size, delete_original=True)
    dp = dp.rand_lights(light_max_saturation=0., light_ambient_range=(1., 1.), light_diffuse_range=(0., 0.),
                        light_specular_range=(0., 0.), light_shininess_range=(40, 40))
    dp = dp.apply_lighting(batch_lighting=1)
    dp = dp.crop_roi_bg(out_size=crop_out_size, delete_original=True)
    dp = dp.apply_bg()  # convert mask_vis back to bool
    dp = dp.gen_coord_2d(width=640, height=480)
    dp = dp.crop_coord_2d(out_size=crop_out_size)
    dp = dp.render_img()
    return dp


def bop_scene_bop_obj_dp(  # scene_src == 2: load pics from bop (cropping based)
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, scene_mode=True, crop_out_size=64,
    vis_ratio_filter_threshold=.5, bbox_zoom_out_ratio=1.5, **kwargs,
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
    dp = dp.crop_roi_bop(out_size=crop_out_size, delete_original=True)
    dp = dp.set_coord_3d()
    return dp


def real_scene_regular_obj_dp(  # scene_src == 3: real exp (adaptive camera intrinsics)
    path=None, obj_list=None, dtype=cc.dtype, device=cc.device, crop_out_size=256, bbox_zoom_out_ratio=1.5,
    real_img_ext='heic', charuco_w_square=7, charuco_h_square=10, charuco_square_length=.04, cam_K=cc.real_cam_K,
    cylinder_scale_true=.042, cylinder_align_x=3., cylinder_align_y=5., sphericon_scale_true=.05, sphericon_align_x=3.,
    sphericon_align_y=5., **kwargs,
):
    assert len(obj_list) == 1
    dp = SampleSource(dtype=dtype, device=device, scene_mode=False, img_render_size=crop_out_size)
    dp = dataloader.datapipe.functional_bop.init_objects(dp, obj_list=obj_list, path=None)
    dp = dp.load_real_scene(path=path, ext=real_img_ext)
    # dp = dp.rand_scene_id()
    dp = dp.set_real_scene()
    dp = dp.set_calib_camera(w_square=charuco_w_square, h_square=charuco_h_square, square_length=charuco_square_length,
                             cam_K=cam_K)
    dp = dp.estimate_pose()
    dp = dp.set_mesh_info()
    dp = dp.offset_pose_cylinder(scale_true=cylinder_scale_true, align_x=cylinder_align_x, align_y=cylinder_align_y)
    dp = dp.offset_pose_sphericon(scale_true=sphericon_scale_true, align_x=sphericon_align_x, align_y=sphericon_align_y)
    dp = dp.gen_bbox_proj()
    dp = dp.dzi_bbox(max_dzi_ratio=0., bbox_zoom_out_ratio=bbox_zoom_out_ratio)
    dp = dp.set_roi_camera()
    dp = dp.render_scene()
    dp = dp.gen_mask()
    dp = dp.compute_vis_ratio()
    dp = dp.crop_roi_real_bg()
    dp = dp.crop_roi_dummy(delete_original=True)
    dp = dp.bg_as_real_img(delete_original=True)
    dp = dp.normalize_normal()
    dp = dp.gen_coord_2d_bbox()
    dp = dp.calibrate_bbox()
    return dp
