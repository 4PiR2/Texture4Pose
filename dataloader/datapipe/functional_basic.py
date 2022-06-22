import os
from typing import Union

import pytorch3d.transforms
import torch
import torch.nn.functional as F
from torch.utils.data import functional_datapipe
import torchvision.transforms as T
import torchvision.transforms.functional as vF

import config.const as cc
from dataloader.obj_mesh import ObjMesh, RegularMesh
from dataloader.datapipe.helper import SampleMapperIDP, SampleFiltererIDP
from dataloader.sample import SampleFields as sf
from renderer.scene import Scene, NHWKC_to_NCHW, NCHW_to_NHWKC
import utils.io
import utils.image_2d
import utils.transform_3d


@functional_datapipe('init_regular_objects')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, obj_list: Union[dict[int, str], list[int]]):
        super().__init__(src_dp, [], [sf.obj_id], required_attributes=['dtype', 'device', 'objects', 'objects_eval'])

        objects = {}
        for obj_id in obj_list:
            objects[obj_id] = RegularMesh(dtype=self.dtype, device=self.device, obj_id=int(obj_id),
                                          name=obj_list[obj_id] if isinstance(obj_list, dict) else None)

        self.objects: dict[int, ObjMesh] = {**self.objects, **objects}
        self.objects_eval: dict[int, ObjMesh] = {**self.objects_eval, **objects}

    def main(self):
        obj_id = torch.tensor(list(self.objects), dtype=torch.uint8, device=self.device)
        return obj_id


@functional_datapipe('set_mesh_infos')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.obj_id], [sf.obj_size, sf.obj_diameter], required_attributes=['objects'])

    def main(self, obj_id: torch.Tensor):
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]
        obj_diameter = torch.tensor([self.objects[int(i)].diameter for i in obj_id])  # [N]
        return obj_size, obj_diameter


@functional_datapipe('set_static_cameras')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, cam_K: torch.Tensor = cc.lm_cam_K):
        super().__init__(src_dp, [sf.N], [sf.cam_K], required_attributes=['dtype', 'device'])
        self._cam_K: torch.Tensor = utils.transform_3d.normalize_cam_K(cam_K.to(self.device, dtype=self.dtype))
        # [3, 3]

    def main(self, N: int):
        return self._cam_K.expand(N, -1, -1)  # [N, 3, 3]


@functional_datapipe('rand_select_objs')
class _(SampleFiltererIDP):
    def __init__(self, src_dp: SampleMapperIDP, num_obj: int = None, repeated_sample_obj: bool = False):
        super().__init__(src_dp, [sf.N], required_attributes=['objects'])
        self._num_obj: int = num_obj if num_obj else len(self.objects)
        self._repeated_sample_obj: bool = repeated_sample_obj

    def main(self, N: int):
        selected, _ = torch.multinomial(torch.ones(N), self._num_obj, replacement=self._repeated_sample_obj).sort()
        return selected


@functional_datapipe('rand_gt_poses')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, random_t_depth_range: tuple[int, int] = (.5, 1.2), cuboid: bool = True):
        super().__init__(src_dp, [sf.obj_id, sf.cam_K], [sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         required_attributes=['objects', 'scene_mode', 'img_render_size'])
        self._random_t_depth_range: tuple[int, int] = random_t_depth_range
        self._cuboid: bool = cuboid

    def main(self, obj_id: torch.Tensor, cam_K: torch.Tensor):
        N = len(obj_id)
        dtype = cam_K.dtype
        device = cam_K.device
        radii = torch.tensor([self.objects[int(oid)].radius for oid in obj_id], dtype=dtype, device=device)  # [N]
        centers = torch.stack([self.objects[int(oid)].center for oid in obj_id], dim=0)  # [N, 3(XYZ)]
        box2d_min = torch.linalg.inv(cam_K)[..., -1]  # [N, 3(XY1)], inv(K) @ [0., 0., 1.].T
        box2d_max = torch.linalg.solve(cam_K, torch.tensor([[self.img_render_size], [self.img_render_size], [1.]],
                                                           dtype=dtype, device=device))[
            ..., 0]  # [N, 3(XY1)], inv(K) @ [W, H, 1.].T
        t_depth_min, t_depth_max = self._random_t_depth_range
        box3d_size = (box2d_max - box2d_min) * t_depth_min - radii[:, None] * 2.  # [N, 3(XYZ)]
        box3d_size[..., -1] += t_depth_max - t_depth_min
        box3d_min = box2d_min * t_depth_min - centers + radii[:, None]  # [N, 3(XYZ)]

        if self.scene_mode:
            triu_indices = torch.triu_indices(N, N, 1)
            mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

        while True:
            gt_cam_t_m2c = torch.rand((N, 3), dtype=dtype, device=device) * box3d_size + box3d_min
            if not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all():
                break
        gt_cam_R_m2c = pytorch3d.transforms.random_rotations(N, dtype=dtype, device=device)  # [N, 3, 3]
        return gt_cam_R_m2c, gt_cam_t_m2c


@functional_datapipe('render_scene')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp,
                         [sf.obj_id, sf.cam_K, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c],
                         [sf.gt_coord_3d, sf.gt_normal, sf.gt_zbuf, sf.gt_texel],
                         required_attributes=['objects', 'img_render_size'])

    def main(self, obj_id: torch.Tensor, cam_K: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor):
        scene = Scene(cam_K, gt_cam_R_m2c, gt_cam_t_m2c, obj_id, self.objects, self.img_render_size,
                      self.img_render_size)
        gt_coord_3d, gt_normal, gt_zbuf, gt_texel = [NHWKC_to_NCHW(x) for x in scene.render_scene()]
        return gt_coord_3d, gt_normal, gt_zbuf, gt_texel


@functional_datapipe('gen_mask')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_zbuf], [sf.gt_mask_vis, sf.gt_mask_obj], [sf.gt_zbuf],
                         required_attributes=['scene_mode'])

    def main(self, gt_zbuf: torch.Tensor):
        gt_mask_obj = gt_zbuf < torch.inf  # [N, 1, H, W]
        if self.scene_mode:
            gt_mask_vis = (gt_zbuf == gt_zbuf.min(dim=0)[0]) * gt_mask_obj  # [N, 1, H, W]
        else:
            gt_mask_vis = gt_mask_obj  # [N, 1, H, W]
        return gt_mask_vis, gt_mask_obj


@functional_datapipe('compute_vis_ratio')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_mask_vis, sf.gt_mask_obj], [sf.gt_vis_ratio], [sf.gt_mask_obj],
                         required_attributes=['dtype', 'scene_mode'])

    def main(self, gt_mask_vis: torch.Tensor, gt_mask_obj: torch.Tensor):
        if self.scene_mode:
            gt_vis_ratio = (gt_mask_vis.bool().sum(dim=(1, 2, 3)) / gt_mask_obj.bool().sum(dim=(1, 2, 3)))\
                .to(dtype=self.dtype)  # [N]
        else:
            gt_vis_ratio = torch.ones(len(gt_mask_vis), dtype=self.dtype, device=gt_mask_vis.device)
        return gt_vis_ratio


@functional_datapipe('filter_vis_ratio')
class _(SampleFiltererIDP):
    def __init__(self, src_dp: SampleMapperIDP, vis_ratio_filter_threshold: float = .5):
        # threshold = 0.: only visible objects
        # threshold = 1.: only un-occluded objects
        super().__init__(src_dp, [sf.gt_vis_ratio], required_attributes=['scene_mode'])
        self._vis_ratio_filter_threshold = vis_ratio_filter_threshold

    def main(self, gt_vis_ratio: torch.Tensor):
        N = len(gt_vis_ratio)
        device = gt_vis_ratio.device
        if self.scene_mode:
            if 0. <= self._vis_ratio_filter_threshold <= 1.:
                if self._vis_ratio_filter_threshold < 1.:
                    mask = gt_vis_ratio > self._vis_ratio_filter_threshold
                else:
                    mask = gt_vis_ratio >= 1.
            elif self._vis_ratio_filter_threshold < 0.:
                mask = torch.ones(N, dtype=torch.bool, device=device)
            else:
                mask = torch.zeros(N, dtype=torch.bool, device=device)
        else:
            if self._vis_ratio_filter_threshold <= 1.:
                mask = torch.ones(N, dtype=torch.bool, device=device)
            else:
                mask = torch.zeros(N, dtype=torch.bool, device=device)
        return mask


@functional_datapipe('gen_bbox')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_mask_vis], [sf.gt_bbox_vis], required_attributes=['dtype'])

    def main(self, gt_mask_vis: torch.Tensor):
        gt_bbox_vis = utils.image_2d.get_bbox2d_from_mask(gt_mask_vis[:, 0]).to(dtype=self.dtype)  # [N, 4(XYWH)]
        return gt_bbox_vis


@functional_datapipe('dzi_bbox')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, max_dzi_ratio: float = .25, bbox_zoom_out_ratio: float = 1.5):
        super().__init__(src_dp, [sf.gt_bbox_vis], [sf.bbox])
        self._max_dzi_ratio: float = max_dzi_ratio
        self._bbox_zoom_out_ratio: float = bbox_zoom_out_ratio

    def main(self, gt_bbox_vis: torch.Tensor):
        dzi_ratio = (torch.rand(len(gt_bbox_vis), 4, dtype=gt_bbox_vis.dtype, device=gt_bbox_vis.device) * 2. - 1.) \
                    * self._max_dzi_ratio
        dzi_ratio[:, 3] = dzi_ratio[:, 2]
        bbox = utils.image_2d.get_dzi_bbox(gt_bbox_vis, dzi_ratio)  # [N, 4(XYWH)]
        bbox[:, 2:] = torch.max(bbox[:, 2:], dim=-1)[0][:, None] * self._bbox_zoom_out_ratio
        return bbox


@functional_datapipe('crop_roi_basic')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, out_size: Union[list[int], int], delete_original: bool = True):
        fields = [sf.gt_mask_vis, sf.gt_coord_3d, sf.gt_normal, sf.gt_texel]
        super().__init__(src_dp, [sf.bbox] + fields, [f'{field}_roi' for field in fields],
                         fields if delete_original else [], required_attributes=['scene_mode'])
        self._out_size: Union[list[int], int] = out_size

    def main(self, bbox: torch.Tensor, gt_mask_vis: torch.Tensor, gt_coord_3d: torch.Tensor, gt_normal: torch.Tensor,
             gt_texel: torch.Tensor):
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self._out_size, mode)

        if self.scene_mode:
            gt_coord_3d = (gt_coord_3d * gt_mask_vis).sum(dim=0)[None]
            gt_normal = (gt_normal * gt_mask_vis).sum(dim=0)[None]
            gt_texel = (gt_texel * gt_mask_vis).sum(dim=0)[None] if gt_texel is not None else None
            gt_mask_vis = (gt_mask_vis * torch.arange(1, len(gt_mask_vis) + 1, dtype=torch.uint8,
                                                      device=gt_mask_vis.device)[:, None, None, None]).sum(dim=0)[None]

        if gt_texel is not None:
            gt_coord_3d_roi, gt_normal_roi, gt_texel_roi = crop([gt_coord_3d, gt_normal, gt_texel], 'bilinear')
        else:
            gt_coord_3d_roi, gt_normal_roi = crop([gt_coord_3d, gt_normal], 'bilinear')
            gt_texel_roi = None
        gt_mask_vis_roi = crop(gt_mask_vis, 'nearest')
        return gt_mask_vis_roi, gt_coord_3d_roi, gt_normal_roi, gt_texel_roi


@functional_datapipe('rand_lights')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, light_color_range=(0., 1.), light_ambient_range=(.5, 1.),
                 light_diffuse_range=(0., .3), light_specular_range=(0., .2), light_shininess_range=(40, 80), ):
        super().__init__(src_dp, [sf.cam_K, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], [sf.o_scene],
                         required_attributes=['scene_mode'])
        self._light_color_range: tuple[float, float] = light_color_range  # \in [0., 1.]
        self._light_ambient_range: tuple[float, float] = light_ambient_range  # \in [0., 1.]
        self._light_diffuse_range: tuple[float, float] = light_diffuse_range  # \in [0., 1.]
        self._light_specular_range: tuple[float, float] = light_specular_range  # \in [0., 1.]
        self._light_shininess_range: tuple[int, int] = light_shininess_range  # \in [0, 1000]

    def main(self, cam_K: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor) -> Scene:
        o_scene = Scene(cam_K, gt_cam_R_m2c, gt_cam_t_m2c)
        N = len(gt_cam_R_m2c)
        B = 1 if self.scene_mode else N
        light_color = torch.rand(B, 3) * (self._light_color_range[1] - self._light_color_range[0]) \
                      + self._light_color_range[0]

        def get_light(intensity_range) -> torch.Tensor:
            light_intensity = torch.rand(B, 1) * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
            return light_intensity * light_color

        direction = torch.randn(B, 3)
        if self.scene_mode:
            direction = (direction.to(gt_cam_R_m2c.device) @ gt_cam_R_m2c)[:, 0]

        shininess = torch.randint(low=self._light_shininess_range[0], high=self._light_shininess_range[1] + 1, size=[N])
        ambient = get_light(self._light_ambient_range)
        diffuse = get_light(self._light_diffuse_range)
        specular = get_light(self._light_specular_range)
        o_scene.set_lights(direction, ambient, diffuse, specular, shininess)
        return o_scene


@functional_datapipe('apply_lighting')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, batch_lighting: int = None):
        super().__init__(src_dp,
                         [sf.o_scene, sf.gt_mask_vis_roi, sf.gt_coord_3d_roi, sf.gt_normal_roi, sf.gt_texel_roi],
                         [sf.gt_light_texel_roi, sf.gt_light_specular_roi], [sf.o_scene, sf.gt_texel_roi],
                         required_attributes=['scene_mode'])
        self._B: int = batch_lighting
        # param batch_lighting is only for batch mode performance optimization, larger -> faster, memory usage is O(N*B)

    def main(self, o_scene: Scene, gt_mask_vis_roi: torch.Tensor, gt_coord_3d_roi: torch.Tensor,
             gt_normal_roi: torch.Tensor, gt_texel_roi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, C, H, W = gt_coord_3d_roi.shape
        if self.scene_mode:
            indices = gt_mask_vis_roi.permute(2, 3, 0, 1).long().expand(1, -1, -1, -1, C)
            # [1, H, W, K(N), C]
            pixel_coords = gt_coord_3d_roi.permute(2, 3, 0, 1).expand(N, -1, -1, -1, -1)  # [N(cam), H, W, N(K), C]
            pixel_normals = gt_normal_roi.permute(2, 3, 0, 1).expand(N, -1, -1, -1, -1)  # [N(cam), H, W, N(K), C]
            B = self._B if self._B else N
            gt_light_texel_roi = []
            gt_light_specular_roi = []
            for i in range(0, N, B):
                light_texel, light_specular = o_scene.apply_lighting(
                    pixel_coords[..., i:i+B, :], pixel_normals[..., i:i+B, :])  # [N(cam), H, W, B(K), C]
                gt_light_texel_roi.append(torch.cat(
                    [torch.zeros_like(light_texel[:1]), light_texel], dim=0
                ).take_along_dim(indices[..., i:i+B, :], dim=0)[0].permute(2, 3, 0, 1))  # [B, C, H, W]
                gt_light_specular_roi.append(torch.cat(
                    [torch.zeros_like(light_specular[:1]), light_specular], dim=0
                ).take_along_dim(indices[..., i:i+B, :], dim=0)[0].permute(2, 3, 0, 1))  # [B, C, H, W]
            gt_light_texel_roi = torch.cat(gt_light_texel_roi, dim=0)
            gt_light_specular_roi = torch.cat(gt_light_specular_roi, dim=0)
        else:
            pixel_coords = NCHW_to_NHWKC(gt_coord_3d_roi)  # [N, H, W, 1(K), C]
            pixel_normals = NCHW_to_NHWKC(gt_normal_roi)  # [N, H, W, 1(K), C]
            light_texel, specular = o_scene.apply_lighting(pixel_coords, pixel_normals)  # [N, H, W, 1(K), C]
            gt_light_texel_roi = NHWKC_to_NCHW(light_texel) * gt_mask_vis_roi  # [N, C, H, W]
            gt_light_specular_roi = NHWKC_to_NCHW(specular) * gt_mask_vis_roi  # [N, C, H, W]
        if gt_texel_roi is not None:
            gt_light_texel_roi *= gt_texel_roi
        return gt_light_texel_roi, gt_light_specular_roi


@functional_datapipe('rand_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp, bg_img_path: str = None):
        super().__init__(src_dp, [sf.N], [sf.gt_bg],
                         required_attributes=['dtype', 'device', 'img_render_size'])
        self._bg_img_path: list[str] = [os.path.join(bg_img_path, name) for name in os.listdir(bg_img_path)] \
            if bg_img_path is not None else None

    def main(self, N: int) -> torch.Tensor:
        if self.scene_mode:
            N = 1
        if self._bg_img_path is not None:
            transform = T.RandomCrop(self.img_render_size)
            idx = torch.randint(len(self._bg_img_path), [N])
            gt_bg = []
            for i in idx:
                im = utils.io.read_img_file(self._bg_img_path[i], dtype=self.dtype, device=self.device)
                im_size = min(im.shape[-2:])
                if im_size < self.img_render_size:
                    im = vF.resize(im, [self.img_render_size])
                im = transform(im)
                gt_bg.append(im)
            gt_bg = torch.cat(gt_bg, dim=0)
        else:
            gt_bg = torch.zeros(1, 3, self.img_render_size, self.img_render_size, dtype=self.dtype, device=self.device)
        return gt_bg


@functional_datapipe('crop_roi_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, out_size: Union[list[int], int], delete_original: bool = True):
        fields = [sf.gt_bg]
        super().__init__(src_dp, [sf.bbox] + fields, [f'{field}_roi' for field in fields],
                         fields if delete_original else [])
        self._out_size: Union[list[int], int] = out_size

    def main(self, bbox: torch.Tensor, gt_bg: torch.Tensor):
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self._out_size, mode)
        return crop(gt_bg, 'bilinear')


@functional_datapipe('apply_bg')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_mask_vis_roi, sf.gt_bg_roi, sf.gt_light_specular_roi],
                         [sf.gt_light_specular_roi, sf.gt_mask_vis_roi], [sf.gt_bg_roi],
                         required_attributes=['scene_mode'])

    def main(self, gt_mask_vis_roi: torch.Tensor, gt_bg_roi: torch.Tensor, gt_light_specular_roi: torch.Tensor):
        gt_light_specular_roi += gt_bg_roi * ~gt_mask_vis_roi.bool()
        if self.scene_mode:
            gt_mask_vis_roi = gt_mask_vis_roi == torch.arange(1, len(gt_mask_vis_roi) + 1, dtype=gt_mask_vis_roi.dtype,
                                                              device=gt_mask_vis_roi.device)[:, None, None, None]
        return gt_light_specular_roi, gt_mask_vis_roi


@functional_datapipe('gen_coord_2d')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, width: int, height: int):
        super().__init__(src_dp, [sf.cam_K], [sf.coord_2d])
        self._width: int = width
        self._height: int = height

    def main(self, cam_K: torch.Tensor):
        if not (cam_K - cam_K[0]).bool().any():
            coord_2d = utils.image_2d.get_coord_2d_map(self._width, self._height, cam_K[0])
        else:
            coord_2d = utils.image_2d.get_coord_2d_map(self._width, self._height, cam_K)
        return coord_2d


@functional_datapipe('crop_coord_2d')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, out_size: Union[list[int], int], delete_original: bool = True):
        fields = [sf.coord_2d]
        super().__init__(src_dp, [sf.bbox] + fields, [f'{field}_roi' for field in fields],
                         fields if delete_original else [])
        self._out_size: Union[list[int], int] = out_size

    def main(self, bbox: torch.Tensor, coord_2d: torch.Tensor):
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self._out_size, mode)
        return crop(coord_2d, 'bilinear')


# @functional_datapipe('render_img')
# class _(SampleMapperIDP):
#     def __init__(self, src_dp):
#         super().__init__(src_dp, [sf.gt_light_texel_roi, sf.gt_light_specular_roi], [sf.img_roi])
#
#     def main(self, gt_light_texel_roi: torch.Tensor, gt_light_specular_roi: torch.Tensor) -> torch.Tensor:
#         img = gt_light_texel_roi + gt_light_specular_roi  # [N, 3(RGB), H, W]
#         return img.clamp(min=0., max=1.)
#
# @functional_datapipe('augment_img')
# class _(SampleMapperIDP):
#     def __init__(self, src_dp, transform=None):
#         super().__init__(src_dp, [sf.img_roi], [sf.img_roi])
#         self._transform = transform
#
#     def main(self, img_roi: torch.Tensor) -> torch.Tensor:
#         if self._transform is not None:
#             img_roi = self._transform(img_roi)
#         return img_roi
