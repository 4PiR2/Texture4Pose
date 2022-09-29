import os
from typing import Union

import pytorch3d.ops
import pytorch3d.transforms
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from torch.utils.data import functional_datapipe
import torchvision.transforms as T
import torchvision.transforms.functional as vF

import config.const as cc
from dataloader.obj_mesh import ObjMesh, RegularMesh
from dataloader.datapipe.helper import SampleMapperIDP, SampleFiltererIDP
from dataloader.sample import SampleFields as sf
from renderer.scene import Scene, NHWKC_to_NCHW, NCHW_to_NHWKC
import utils.color
import utils.io
import utils.image_2d
import utils.transform_3d


@functional_datapipe('init_regular_objects')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, obj_list: Union[dict[int, str], list[int]]):
        super().__init__(src_dp, [], [sf.obj_id], required_attributes=['dtype', 'device', 'objects', 'objects_eval'])

        objects = {}
        objects_eval = {}
        for obj_id in obj_list:
            objects[obj_id] = RegularMesh(dtype=self.dtype, device=self.device, obj_id=int(obj_id),
                                          name=obj_list[obj_id] if isinstance(obj_list, dict) else None, level=5)
            mesh = objects[obj_id].mesh
            objects_eval[obj_id] = Meshes(
                verts=pytorch3d.ops.sample_points_from_meshes(meshes=mesh, num_samples=len(mesh.verts_packed())),
                faces=torch.empty(1, 0, 3, dtype=torch.int, device=self.device)
            )

        self.objects: dict[int, ObjMesh] = {**self.objects, **objects}
        self.objects_eval: dict[int, ObjMesh] = {**self.objects_eval, **objects_eval}

    def main(self):
        obj_id = torch.tensor(list(self.objects), dtype=torch.uint8, device=self.device)
        return obj_id


@functional_datapipe('set_mesh_info')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.obj_id], [sf.obj_size, sf.obj_diameter], required_attributes=['objects'])

    def main(self, obj_id: torch.Tensor):
        obj_size = torch.stack([self.objects[int(i)].size for i in obj_id], dim=0)  # extents: [N, 3(XYZ)]
        obj_diameter = torch.tensor([self.objects[int(i)].diameter for i in obj_id])  # [N]
        return obj_size, obj_diameter


@functional_datapipe('set_static_camera')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, cam_K: torch.Tensor = torch.eye(3), orig: bool = False):
        super().__init__(src_dp, [sf.N], [sf.cam_K] if not orig else [sf.cam_K_orig],
                         required_attributes=['dtype', 'device'])
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


@functional_datapipe('rand_gt_translation_inside_camera')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, random_t_depth_range: tuple[float, float] = (.5, 1.2),
                 cuboid: bool = False):
        super().__init__(src_dp, [sf.obj_id, sf.cam_K], [sf.gt_cam_t_m2c],
                         required_attributes=['objects', 'scene_mode', 'img_render_size'])
        self._random_t_depth_range: tuple[float, float] = random_t_depth_range
        self._cuboid: bool = cuboid

    def main(self, obj_id: torch.Tensor, cam_K: torch.Tensor) -> torch.Tensor:
        N = len(obj_id)
        dtype = cam_K.dtype
        device = cam_K.device
        radii = torch.tensor([self.objects[int(oid)].radius for oid in obj_id], dtype=dtype, device=device)  # [N]

        if self.scene_mode:
            triu_indices = torch.triu_indices(N, N, 1)
            mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

        box2d_min = torch.linalg.inv(cam_K)[..., -1]  # [N, 3(XY1)], inv(K) @ [0., 0., 1.].T
        box2d_max = torch.linalg.solve(cam_K, torch.tensor([[self.img_render_size], [self.img_render_size], [1.]],
            dtype=dtype, device=device))[..., 0]  # [N, 3(XY1)], inv(K) @ [W, H, 1.].T
        t_depth_min, t_depth_max = self._random_t_depth_range
        if self._cuboid:
            centers = torch.stack([self.objects[int(oid)].center for oid in obj_id], dim=0)  # [N, 3(XYZ)]
            box3d_size = (box2d_max - box2d_min) * t_depth_min - radii[..., None] * 2.  # [N, 3(XYZ)]
            box3d_size[..., -1] += t_depth_max - t_depth_min
            box3d_min = box2d_min * t_depth_min - centers + radii[..., None]  # [N, 3(XYZ)]
            while True:
                gt_cam_t_m2c = torch.rand((N, 3), dtype=dtype, device=device) * box3d_size + box3d_min
                if not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all():
                    break
        else:
            box2d = torch.cat([box2d_min[..., :-1], box2d_max[..., :-1]], dim=-1)
            while True:
                tz = torch.rand(N, 1, dtype=dtype, device=device)  # [N, 1]
                tz = (tz * t_depth_max ** 3 + (1. - tz) * t_depth_min ** 3) ** (1. / 3.)  # uniform
                dist_depth = torch.stack([tz[..., -1] - t_depth_min, t_depth_max - tz[..., -1]], dim=-1)  # [N, 2]
                if (dist_depth < radii[..., None]).any():
                    continue
                txy = torch.rand(N, 2, dtype=dtype, device=device) * self.img_render_size  # [N, 2]
                gt_cam_t_m2c = torch.linalg.solve(cam_K,
                    torch.cat([txy, torch.ones_like(txy[..., :1])], dim=-1)[..., None])[..., 0] * tz  # [N, 3]
                dist_box2d = (box2d * gt_cam_t_m2c[..., -1:] - gt_cam_t_m2c[..., :-1].repeat(1, 2)).abs() \
                             / (box2d ** 2 + 1.) ** .5  # [N, 4]
                if (dist_box2d >= radii[..., None]).all() and \
                        (not self.scene_mode or (F.pdist(gt_cam_t_m2c) >= mdist).all()):
                    break
        return gt_cam_t_m2c


@functional_datapipe('rand_gt_rotation')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N], [sf.gt_cam_R_m2c], required_attributes=['dtype', 'device'])

    def main(self, N: int):
        return pytorch3d.transforms.random_rotations(N, dtype=self.dtype, device=self.device)  # [N, 3, 3]


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
        super().__init__(src_dp, [sf.gt_mask_vis, sf.gt_mask_obj], [sf.gt_vis_ratio],  # [sf.gt_mask_obj],
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
        fields = [sf.gt_mask_vis, sf.gt_mask_obj, sf.gt_coord_3d, sf.gt_normal, sf.gt_texel]
        super().__init__(src_dp, [sf.bbox] + fields, [f'{field}_roi' for field in fields],
                         fields if delete_original else [], required_attributes=['scene_mode'])
        self._out_size: Union[list[int], int] = out_size

    def main(self, bbox: torch.Tensor, gt_mask_vis: torch.Tensor, gt_mask_obj: torch.Tensor, gt_coord_3d: torch.Tensor,
             gt_normal: torch.Tensor, gt_texel: torch.Tensor):
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        crop = lambda img, mode: utils.image_2d.crop_roi(img, bbox, crop_size, self._out_size, mode) \
            if img is not None else None

        if self.scene_mode:
            gt_coord_3d = (gt_coord_3d * gt_mask_vis).sum(dim=0)[None]
            gt_normal = (gt_normal * gt_mask_vis).sum(dim=0)[None]
            gt_texel = (gt_texel * gt_mask_vis).sum(dim=0)[None] if gt_texel is not None else None
            gt_mask_vis = (gt_mask_vis * torch.arange(1, len(gt_mask_vis) + 1, dtype=torch.uint8,
                                                      device=gt_mask_vis.device)[:, None, None, None]).sum(dim=0)[None]

        gt_mask_vis_roi = crop(gt_mask_vis, 'nearest')
        gt_mask_obj_roi = crop(gt_mask_obj, 'nearest')
        gt_coord_3d_roi = crop(gt_coord_3d, 'bilinear')
        gt_normal_roi = crop(gt_normal, 'bilinear')
        gt_texel_roi = crop(gt_texel, 'bilinear')
        return gt_mask_vis_roi, gt_mask_obj_roi, gt_coord_3d_roi, gt_normal_roi, gt_texel_roi


@functional_datapipe('rand_occlude')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, occlusion_size_min: float = .125, occlusion_size_max: float = .5,
                 num_occlusion_per_obj: int = 2, min_occlusion_vis_ratio: float = .5, batch_occlusion: int = None):
        super().__init__(src_dp, [sf.gt_mask_vis_roi], [sf.gt_mask_vis_roi])
        # rectangular occlusion
        self._occlusion_size_min: float = occlusion_size_min
        self._occlusion_size_max: float = occlusion_size_max
        self._num_occlusion_per_obj: int = num_occlusion_per_obj
        self._min_occlusion_vis_ratio: float = min_occlusion_vis_ratio
        self._B: int = batch_occlusion

    def main(self, gt_mask_vis_roi: torch.Tensor):
        if self._occlusion_size_max <= 0.:
            return gt_mask_vis_roi
        N, _, H, W = gt_mask_vis_roi.shape
        vis_count = gt_mask_vis_roi.bool().sum(dim=[-3, -2, -1])
        gt_mask_vis_roi_occ = torch.empty_like(gt_mask_vis_roi)
        B = self._B if self._B else N
        for i in range(0, N, B):
            while True:
                gt_mask_vis_roi_occ[i:i + B] = gt_mask_vis_roi[i:i + B]
                for _ in range(self._num_occlusion_per_obj):
                    wh = torch.rand(B, 2) * (self._occlusion_size_max - self._occlusion_size_min) \
                         + self._occlusion_size_min
                    x0y0 = torch.rand(B, 2) * (1. - wh)
                    x0y0wh = (torch.cat([x0y0, wh], dim=-1) * torch.tensor([W, H] * 2)).round().int()
                    for j in range(B):
                        x0, y0, w, h = x0y0wh[j]
                        gt_mask_vis_roi_occ[i + j, :, y0:y0 + h, x0:x0 + w] = 0
                vis_ratio = gt_mask_vis_roi_occ[i:i + B].bool().sum(dim=[-3, -2, -1]) / vis_count[i:i + B]
                if (vis_ratio > self._min_occlusion_vis_ratio).all():
                    break
        return gt_mask_vis_roi_occ


@functional_datapipe('rand_lights')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, light_max_saturation=1., light_ambient_range=(.5, 1.),
                 light_diffuse_range=(0., .3), light_specular_range=(0., .2), light_shininess_range=(40, 80), ):
        super().__init__(src_dp, [sf.cam_K, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], [sf.o_scene],
                         required_attributes=['scene_mode'])
        self._light_max_saturation: float = light_max_saturation  # \in [0., 1.]
        self._light_ambient_range: tuple[float, float] = light_ambient_range  # \in [0., 1.]
        self._light_diffuse_range: tuple[float, float] = light_diffuse_range  # \in [0., 1.]
        self._light_specular_range: tuple[float, float] = light_specular_range  # \in [0., 1.]
        self._light_shininess_range: tuple[int, int] = light_shininess_range  # \in [0, 1000]

    def main(self, cam_K: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor) -> Scene:
        o_scene = Scene(cam_K, gt_cam_R_m2c, gt_cam_t_m2c)
        N = len(gt_cam_R_m2c)
        B = 1 if self.scene_mode else N
        light_color = utils.color.random_color_v_eq_1(B, self._light_max_saturation)

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


@functional_datapipe('render_img')
class _(SampleMapperIDP):
    def __init__(self, src_dp):
        super().__init__(src_dp, [sf.gt_light_texel_roi, sf.gt_light_specular_roi], [sf.img_roi])

    def main(self, gt_light_texel_roi: torch.Tensor, gt_light_specular_roi: torch.Tensor) -> torch.Tensor:
        img_roi = gt_light_texel_roi + gt_light_specular_roi  # [N, 3(RGB), H, W]
        return img_roi.clamp(min=0., max=1.)


@functional_datapipe('augment_img')
class _(SampleMapperIDP):
    def __init__(self, src_dp, transform=None):
        super().__init__(src_dp, [sf.img_roi], [sf.img_roi])
        self._transform = transform

    def main(self, img_roi: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            img_roi = self._transform(img_roi)
        return img_roi


@functional_datapipe('rand_gt_translation')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, random_t_depth_range: tuple[float, float] = (.5, 1.2),
                 random_t_center_range: tuple[float, float] = (-.7, .7), cuboid: bool = False):
        super().__init__(src_dp, [sf.N], [sf.gt_cam_t_m2c],
                         required_attributes=['dtype', 'device', 'objects'])
        self._random_t_depth_range: tuple[float, float] = random_t_depth_range
        self._random_t_center_range: tuple[float, float] = random_t_center_range
        self._cuboid: bool = cuboid

    def main(self, N: int) -> torch.Tensor:
        t_depth_min, t_depth_max = self._random_t_depth_range
        t_center_min, t_center_max = self._random_t_center_range
        if self._cuboid:
            tmin = torch.tensor([t_center_min, t_center_min, t_depth_min], dtype=self.dtype, device=self.device)
            tmax = torch.tensor([t_center_max, t_center_max, t_depth_max], dtype=self.dtype, device=self.device)
            gt_cam_t_m2c = torch.lerp(tmin, tmax, torch.rand((N, 3), dtype=self.dtype, device=self.device))
        else:
            # frustum
            tz = torch.rand(N, 1, dtype=self.dtype, device=self.device)  # [N, 1(Z)]
            tz = (tz * t_depth_max ** 3 + (1. - tz) * t_depth_min ** 3) ** (1. / 3.)  # uniform
            txy = torch.lerp(torch.tensor(t_center_min, dtype=self.dtype, device=self.device),
                             torch.tensor(t_center_max, dtype=self.dtype, device=self.device),
                             torch.rand(N, 2, dtype=self.dtype, device=self.device))  # [N, 2(XY)]
            gt_cam_t_m2c = torch.cat([txy * tz, tz], dim=-1)
        return gt_cam_t_m2c


@functional_datapipe('gen_bbox_proj')
class _(SampleMapperIDP):
    def __init__(self, src_dp):
        super().__init__(src_dp, [sf.obj_id, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c], [sf.gt_bbox_vis],
                         required_attributes=['objects'])

    def main(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor) -> torch.Tensor:
        bbox = []
        for oid, cam_R, cam_t in zip(obj_id, gt_cam_R_m2c, gt_cam_t_m2c):
            mesh = self.objects[int(oid)].mesh
            verts = mesh.verts_packed() @ cam_R.transpose(-2, -1) + cam_t
            pverts = verts[..., :2] / verts[..., -1:]
            pmin, _ = pverts.min(dim=-2)
            pmax, _ = pverts.max(dim=-2)
            bbox.append(torch.cat([(pmin + pmax) * .5, pmax - pmin], dim=-1))
        return torch.stack(bbox, dim=0)


@functional_datapipe('set_roi_camera')
class _(SampleMapperIDP):
    def __init__(self, src_dp):
        super().__init__(src_dp, [sf.bbox], [sf.cam_K])

    def main(self, bbox: torch.Tensor) -> torch.Tensor:
        cam_K = torch.zeros(len(bbox), 3, 3, dtype=bbox.dtype, device=bbox.device)
        cam_K[..., 2, 2] = 1.
        cam_K[..., 0, 0] = cam_K[..., 1, 1] = bbox[..., 2:].max(dim=-1)[0] / 256.
        cam_K[..., :2, 2] = bbox[..., :2] - bbox[..., 2:] * .5
        return torch.linalg.inv(cam_K)


@functional_datapipe('crop_roi_dummy')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, delete_original: bool = True):
        fields = [sf.gt_mask_vis, sf.gt_mask_obj, sf.gt_coord_3d, sf.gt_normal, sf.gt_texel, sf.gt_bg]
        super().__init__(src_dp, fields, [f'{field}_roi' for field in fields], fields if delete_original else [])

    def main(self, **kwargs):
        return kwargs.values()


@functional_datapipe('normalize_normal')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.gt_normal_roi], [sf.gt_normal_roi])

    def main(self, gt_normal_roi) -> torch.Tensor:
        return F.normalize(gt_normal_roi, p=2, dim=-3)


@functional_datapipe('gen_coord_2d_bbox')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.bbox], [sf.coord_2d_roi], required_attributes=['dtype', 'device', 'img_render_size'])
        self._coord_2d = utils.image_2d.get_coord_2d_map(self.img_render_size, self.img_render_size, dtype=self.dtype,
                                                         device=self.device) / self.img_render_size  # [2(XY), H, W]

    def main(self, bbox: torch.Tensor) -> torch.Tensor:
        pmin = bbox[:, :2] - bbox[:, 2:] * .5
        pmax = bbox[:, :2] + bbox[:, 2:] * .5
        return torch.lerp(pmin[..., None, None], pmax[..., None, None], self._coord_2d)


@functional_datapipe('calibrate_bbox')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.N], [sf.bbox], required_attributes=['dtype', 'device', 'img_render_size'])

    def main(self, N: int) -> torch.Tensor:
        w = self.img_render_size
        x = w * .5
        return torch.tensor([x, x, w, w], dtype=self.dtype, device=self.device).expand(N, -1)
