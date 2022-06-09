from typing import Any, Union, Callable

import torch
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, \
    DirectionalLights, Materials
from pytorch3d.renderer.mesh import TexturesBase
from pytorch3d.structures import join_meshes_as_batch, Meshes

from dataloader.obj_mesh import ObjMesh
from renderer.gt_shader import GTHardPhongShader
from utils.image_2d import get_bbox2d_from_mask
from utils.transform_3d import normalize_cam_K


class Scene:
    texture_net: Callable[[ObjMesh], TexturesBase] = None

    def __init__(self, objects: dict[int, ObjMesh], cam_K: torch.Tensor, obj_id: torch.Tensor,
                 gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor, width: int = 512, height: int = 512, **kwargs):
        self.kwargs: dict[str, Any] = kwargs
        self.objects: dict[int, ObjMesh] = objects  # dict
        self.device: torch.device = gt_cam_R_m2c.device
        self.dtype: torch.dtype = gt_cam_R_m2c.dtype
        self.cam_K: torch.Tensor = normalize_cam_K(cam_K)  # [N, 3, 3]
        self.obj_id: torch.Tensor = obj_id  # [N]
        self.gt_cam_R_m2c: torch.Tensor = gt_cam_R_m2c  # [N, 3, 3]
        self.gt_cam_t_m2c: torch.Tensor = gt_cam_t_m2c  # [N, 3]
        self.width: int = width
        self.height: int = height

        self.cameras = self._get_cameras()
        self.obj_meshes = self._get_obj_meshes()
        self.rasterizer = self._get_rasterizer()

        self.gt_coord_3d_vis, self.gt_coord_3d_obj = None, None
        self.gt_mask, self.gt_mask_vis, self.gt_mask_obj = None, None, None
        self.gt_vis_ratio = None
        self.gt_bbox_vis, self.gt_bbox_obj = None, None

    def _get_cameras(self) -> PerspectiveCameras:
        dtype = torch.float32
        K = torch.zeros(len(self.cam_K), 4, 4, dtype=dtype, device=self.device)
        K[:, :2, :3] = self.cam_K[:, :2]
        K[:, 2, 3] = K[:, 3, 2] = 1.
        # ref: https://github.com/wangg12/pytorch3d_render_linemod
        R = self.gt_cam_R_m2c.clone().to(dtype=dtype).transpose(-2, -1)
        R[..., :2] *= -1.
        t = self.gt_cam_t_m2c.clone().to(dtype=dtype)
        t[..., :2] *= -1.
        return PerspectiveCameras(R=R, T=t, K=K, image_size=((self.height, self.width),), device=self.device,
                                  in_ndc=False)

    def _get_obj_meshes(self) -> list[Meshes]:
        unique_ids, inverse_indices = self.obj_id.unique(return_inverse=True)
        meshes = [self.objects[int(uid)].mesh.clone() for uid in unique_ids]
        return [meshes[idx] for idx in inverse_indices]

    def _set_obj_textures(self, func_get_texture: Callable[[ObjMesh], TexturesBase]) -> None:
        unique_ids, inverse_indices = self.obj_id.unique(return_inverse=True)
        textures = []
        for uid in unique_ids:
            textures.append(self.objects[int(uid)].get_texture(func_get_texture))
        for m, i in zip(self.obj_meshes, inverse_indices):
            m.textures = textures[i]

    def _get_rasterizer(self) -> MeshRasterizer:
        rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=RasterizationSettings(
                image_size=(self.height, self.width),
                blur_radius=0.,
                faces_per_pixel=1,
                max_faces_per_bin=max([int(mesh.num_faces_per_mesh().max()) for mesh in self.obj_meshes]),
                # bin_size=0,  # Noisy Renderings on LM: https://github.com/facebookresearch/pytorch3d/issues/867
            ),
        )
        return rasterizer

    def render_scene(self, f: Callable[[ObjMesh], TexturesBase] = None,
                     ambient: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((1.,) * 3,),
                     diffuse: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                     specular: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                     direction: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0., 0., -1.),),
                     shininess: Union[torch.Tensor, int] = 64) -> torch.Tensor:
        lights = DirectionalLights(ambient_color=ambient, diffuse_color=diffuse, specular_color=specular,
                                   direction=direction, device=self.device)
        materials = Materials(shininess=shininess, device=self.device)

        renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=GTHardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights,
                materials=materials,
                # blend_params=None,
            )
        )

        self._set_obj_textures(Scene.texture_net if f is None else f)

        scene_mesh = self._get_scene_meshes()

        images, texels, mask_obj, mask_vis, pixel_coords, pixel_normals = renderer(scene_mesh, include_textures=True)
        images = images.to(dtype=self.dtype)
        texels = texels.to(dtype=self.dtype)
        pixel_coords = pixel_coords.to(dtype=self.dtype)
        pixel_normals = pixel_normals.to(dtype=self.dtype)

        self.images = images.clamp(min=0., max=1.)
        self.gt_coord_3d_vis, self.gt_coord_3d_obj = pixel_coords * mask_vis, pixel_coords
        self.gt_mask_vis, self.gt_mask_obj = mask_vis, mask_obj
        self.gt_bbox_obj = get_bbox2d_from_mask(self.gt_mask_obj[:, 0]).to(dtype=self.dtype)  # [N, 4(XYWH)]

        self._post_process(texels, mask_obj, mask_vis, pixel_coords, pixel_normals)
        return self.images

    def _get_scene_meshes(self) -> Meshes:
        return join_meshes_as_batch(self.obj_meshes, include_textures=True)

    def _post_process(self, texels, mask_obj, mask_vis, pixel_coords, pixel_normals):
        self.images = (self.images * mask_vis).sum(dim=0)[None]
        self.gt_mask = self.gt_mask_obj.any(dim=0)[None]  # [1, 1, H, W]
        self.gt_vis_ratio = self.gt_mask_vis.sum(dim=(1, 2, 3)) / self.gt_mask_obj.sum(dim=(1, 2, 3))  # [N]
        self.gt_bbox_vis = get_bbox2d_from_mask(self.gt_mask_vis[:, 0]).to(dtype=self.dtype)  # [N, 4(XYWH)]


class SceneBatch(Scene):
    def _post_process(self, texels, mask_obj, mask_vis, pixel_coords, pixel_normals):
        self.gt_mask = self.gt_mask_vis = self.gt_mask_obj  # [N, 1, H, W]
        self.gt_vis_ratio = torch.ones(len(self.gt_cam_R_m2c), dtype=self.dtype, device=self.device)
        self.gt_bbox_vis = self.gt_bbox_obj  # [N, 4(XYWH)]


class SceneBatchOne(SceneBatch):
    def _get_obj_meshes(self) -> list[Meshes]:
        return [self.objects[int(self.obj_id[0])].mesh.clone()]

    def _set_obj_textures(self, func_get_texture: Callable[[ObjMesh], TexturesBase]) -> None:
        self.obj_meshes[0].textures = self.objects[int(self.obj_id[0])].get_texture(func_get_texture)

    def _get_scene_meshes(self) -> Meshes:
        return self.obj_meshes[0].extend(len(self.gt_cam_R_m2c))
