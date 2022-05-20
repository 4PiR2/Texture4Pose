from typing import Any, Union, Callable

import torch
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    MeshRenderer, SoftPhongShader, BlendParams, DirectionalLights, Materials, HardPhongShader, TexturesVertex
from pytorch3d.renderer.mesh import TexturesBase
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch, Meshes

from dataloader.obj_mesh import ObjMesh
from utils.simple_shader import SimpleShader
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
        self._set_obj_textures(self._get_func_get_texture_gt())
        self.rasterizer = self._get_rasterizer()
        gt_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SimpleShader())
        gt_images = gt_renderer(self._get_gt_scene_meshes()).permute(0, 3, 1, 2)
        # [1+N, H, W, 4(IXYZ)] -> [1+N, 4(IXYZ), H, W]

        self.gt_coord_3d_vis, self.gt_coord_3d_obj = self._get_coord_3d(gt_images)
        self.gt_mask, self.gt_mask_vis, self.gt_mask_obj = self._get_mask(gt_images)
        self.gt_vis_ratio = self._get_vis_ratio()
        self.gt_bbox_vis, self.gt_bbox_obj = self._get_bboxes()

    def _get_cameras(self) -> PerspectiveCameras:
        R = torch.eye(3, dtype=self.dtype, device=self.device)[None]
        R[0, 0, 0] = R[0, 1, 1] = -1.
        return PerspectiveCameras(R=R, K=self._get_cam_K()[:1], image_size=((self.height, self.width),), device=self.device,
                                  in_ndc=False)

    def _get_cam_K(self) -> torch.Tensor:
        K = torch.zeros(len(self.cam_K), 4, 4, dtype=self.dtype, device=self.device)
        K[:, :2, :3] = self.cam_K[:, :2]
        K[:, 2, 3] = K[:, 3, 2] = 1.
        return K

    def _get_obj_meshes(self) -> list[Meshes]:
        return [self.objects[int(self.obj_id[i])].get_transformed_mesh(self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i])
                for i in range(len(self.obj_id))]

    @staticmethod
    def _get_func_get_texture_gt() -> Callable[[ObjMesh], TexturesBase]:
        def f(obj: ObjMesh):
            verts = obj.mesh.verts_packed()  # [V, 3(XYZ)]
            gt_texture = torch.cat([torch.full((len(verts), 1), obj.obj_id, dtype=verts.dtype, device=verts.device),
                                    verts], dim=1)
            return TexturesVertex(gt_texture[None])  # [1, V, 4(IXYZ)]

        return f

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

    def _get_gt_scene_meshes(self) -> Meshes:
        scene_mesh = join_meshes_as_scene(self.obj_meshes, include_textures=True)
        return join_meshes_as_batch([scene_mesh, *self.obj_meshes], include_textures=True)

    @staticmethod
    def _get_coord_3d(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coord_3d = images[:, 1:4]  # [1+N, 3(XYZ), H, W]
        gt_coord_3d_vis = coord_3d[:1]  # [1, 3(XYZ), H, W]
        gt_coord_3d_obj = coord_3d[1:]  # [N, 3(XYZ), H, W]
        return gt_coord_3d_vis, gt_coord_3d_obj

    def _get_mask(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = images[:, :1].round().to(dtype=torch.uint8)  # [1+N, 1, H, W]
        gt_mask = masks[:1].bool()  # [1, 1, H, W]
        gt_mask_vis = torch.cat([masks[:1] == obj_id for obj_id in self.obj_id], dim=0)  # [N, 1, H, W]
        gt_mask_obj = masks[1:].bool()  # [N, 1, H, W]
        return gt_mask, gt_mask_vis, gt_mask_obj

    def _get_vis_ratio(self) -> torch.Tensor:
        return self.gt_mask_vis.sum(dim=(1, 2, 3)) / self.gt_mask_obj.sum(dim=(1, 2, 3))  # [N]

    def _get_bboxes(self) -> tuple[torch.Tensor, torch.Tensor]:
        gt_bbox_vis = get_bbox2d_from_mask(self.gt_mask_vis)[:, 0]  # [N, 4(XYWH)]
        gt_bbox_obj = get_bbox2d_from_mask(self.gt_mask_obj)[:, 0]  # [N, 4(XYWH)]
        return gt_bbox_vis, gt_bbox_obj

    def render_scene(self, f: Callable[[ObjMesh], TexturesBase] = None,
                     ambient: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((1.,) * 3,),
                     diffuse: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                     specular: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                     direction: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0., 0., -1.),),
                     shininess: Union[torch.Tensor, int] = 64) -> torch.Tensor:
        lights = DirectionalLights(ambient_color=ambient, diffuse_color=diffuse, specular_color=specular,
                                   direction=direction, device=self.device)
        materials = Materials(shininess=shininess, device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.,) * 3)

        renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights,
                materials=materials,
                blend_params=blend_params,
            )
        )

        self._set_obj_textures(Scene.texture_net if f is None else f)

        scene_mesh = self._get_scene_meshes()

        images = renderer(scene_mesh, include_textures=True).permute(0, 3, 1, 2)[:, :3]  # [B, 3(RGB), H, W]
        images = images.clamp(min=0., max=1.)
        return images

    def _get_scene_meshes(self) -> Meshes:
        return join_meshes_as_scene(self.obj_meshes, include_textures=True)


class SceneBatch(Scene):
    def _get_cameras(self) -> PerspectiveCameras:
        # ref: https://github.com/wangg12/pytorch3d_render_linemod
        R = self.gt_cam_R_m2c.clone().transpose(-2, -1)
        R[..., :2] *= -1.
        t = self.gt_cam_t_m2c.clone()
        t[..., :2] *= -1.
        return PerspectiveCameras(R=R, T=t, K=self._get_cam_K(), image_size=((self.height, self.width),),
                                  device=self.device, in_ndc=False)

    def _get_obj_meshes(self) -> list[Meshes]:
        unique_ids, inverse_indices = self.obj_id.unique(return_inverse=True)
        meshes = [self.objects[int(uid)].mesh.clone() for uid in unique_ids]
        return [meshes[idx] for idx in inverse_indices]

    def _get_gt_scene_meshes(self) -> Meshes:
        return join_meshes_as_batch(self.obj_meshes, include_textures=True)

    @staticmethod
    def _get_coord_3d(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coord_3d = images[:, 1:4]  # [N, 3(XYZ), H, W]
        return coord_3d, coord_3d

    def _get_mask(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = images[:, :1].bool()  # [N, 1, H, W]
        return mask, mask, mask

    def _get_vis_ratio(self) -> torch.Tensor:
        return torch.ones(len(self.gt_cam_R_m2c), dtype=self.dtype, device=self.device)

    def _get_bboxes(self) -> tuple[torch.Tensor, torch.Tensor]:
        gt_bbox_obj = get_bbox2d_from_mask(self.gt_mask_obj)[:, 0]  # [N, 4(XYWH)]
        return gt_bbox_obj, gt_bbox_obj

    def _get_scene_meshes(self) -> Meshes:
        return join_meshes_as_batch(self.obj_meshes, include_textures=True)


class SceneBatchOne(SceneBatch):
    def _get_obj_meshes(self) -> list[Meshes]:
        return [self.objects[int(self.obj_id[0])].mesh.clone()]

    def _set_obj_textures(self, func_get_texture: Callable[[ObjMesh], TexturesBase]) -> None:
        self.obj_meshes[0].textures = self.objects[int(self.obj_id[0])].get_texture(func_get_texture)

    def _get_gt_scene_meshes(self) -> Meshes:
        return self._get_scene_meshes()

    def _get_scene_meshes(self) -> Meshes:
        return self.obj_meshes[0].extend(len(self.gt_cam_R_m2c))
