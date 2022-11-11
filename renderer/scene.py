from typing import Callable, Optional, Union

from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, Materials, \
    TensorProperties
from pytorch3d.renderer.mesh import TexturesBase
from pytorch3d.renderer.mesh.shading import _apply_lighting
from pytorch3d.structures import join_meshes_as_batch
import torch
from torch.nn import functional as F

from dataloader.obj_mesh import ObjMesh
from renderer.gt_shader import GTHardPhongShader
from renderer.lighting import DirectionalLights
from utils.transform_3d import normalize_cam_K


def NHWKC_to_NCHW(x: torch.Tensor):
    """

    :param x: [N, H, W, 1(K), C] or [N, H, W, 1(K)]
    :return: [N, C, H, W]
    """
    if x is None:
        return None
    N, H, W, K, *C = x.shape
    assert K == 1
    if C:
        return x[..., 0, :].permute(0, 3, 1, 2)  # [N, C, H, W]
    else:
        return x[:, None, ..., 0]  # [N, 1(C), H, W]


def NCHW_to_NHWKC(x: torch.Tensor):
    """

    :param x: [N, C, H, W]
    :return: [N, H, W, 1(K), C] or [N, H, W, 1(K)]
    """
    if x is None:
        return None
    N, C, H, W = x.shape
    if C > 1:
        return x.permute(0, 2, 3, 1)[..., None, :]  # [N, H, W, 1(K), C]
    else:
        return x[:, 0, ..., None]  # [N, H, W, 1(K)]


class Scene:
    texture_net_v: Callable[[ObjMesh], TexturesBase] = None
    texture_net_p: Callable = None

    def __init__(self, cam_K: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor,
                 obj_id: torch.Tensor = None, objects: dict[int, ObjMesh] = None, width: int = 512, height: int = 512):
        self.device: torch.device = gt_cam_R_m2c.device
        self.dtype: torch.dtype = gt_cam_R_m2c.dtype
        self.obj_id: torch.Tensor = obj_id  # [N]
        self.objects: dict[int, ObjMesh] = objects  # dict
        self.width: int = width
        self.height: int = height
        cam_K: torch.Tensor = normalize_cam_K(cam_K)  # [N, 3, 3]

        dtype = torch.float32
        K = torch.zeros(len(cam_K), 4, 4, dtype=dtype, device=self.device)
        K[:, :2, :3] = cam_K[:, :2]
        K[:, 2, 3] = K[:, 3, 2] = 1.
        # ref: https://github.com/wangg12/pytorch3d_render_linemod
        R = gt_cam_R_m2c.to(dtype=dtype).clone().transpose(-2, -1)
        R[..., :2] *= -1.
        t = gt_cam_t_m2c.to(dtype=dtype).clone()
        t[..., :2] *= -1.
        self.cameras = PerspectiveCameras(R=R, T=t, K=K, image_size=((height, width),), device=self.device,
                                          in_ndc=False)

        self.lights: Optional[TensorProperties] = None
        self.materials: Optional[Materials] = None

    def render_scene(self, f: Callable[[ObjMesh], TexturesBase] = None):
        unique_ids, inverse_indices = self.obj_id.unique(return_inverse=True)
        meshes = [self.objects[int(uid)].mesh.clone() for uid in unique_ids]
        obj_meshes = [meshes[idx] for idx in inverse_indices]

        rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=RasterizationSettings(
                image_size=(self.height, self.width),
                blur_radius=0.,
                faces_per_pixel=1,
                max_faces_per_bin=max([int(mesh.num_faces_per_mesh().max()) for mesh in obj_meshes]),
                # bin_size=0,  # Noisy Renderings on LM: https://github.com/facebookresearch/pytorch3d/issues/867
                cull_backfaces=True,
            ),
        )

        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=GTHardPhongShader(
                device=self.device,
                dtype=self.dtype,
            )
        )

        unique_ids, inverse_indices = self.obj_id.unique(return_inverse=True)
        textures = []
        include_textures = Scene.texture_net_p is None
        for uid in unique_ids:
            t = self.objects[int(uid)].get_texture(Scene.texture_net_v if f is None else f)
            textures.append(t)
            if t is None:
                include_textures = False
        for m, i in zip(obj_meshes, inverse_indices):
            m.textures = textures[i]

        scene_mesh = join_meshes_as_batch(obj_meshes, include_textures=include_textures)
        pixel_coords, pixel_normals, zbuf, texels = renderer(scene_mesh)
        return pixel_coords, pixel_normals, zbuf, texels

    def set_lights(self, direction: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0., 0., -1.),),
                         ambient: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((1.,) * 3,),
                         diffuse: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                         specular: Union[torch.Tensor, tuple[tuple[float, float, float]]] = ((0.,) * 3,),
                         shininess: Union[torch.Tensor, int] = 64) -> None:
        """

        :param direction: [N, 3(XYZ)], un-normalized
        :param ambient: [N, 3(RGB)]
        :param diffuse: [N, 3(RGB)]
        :param specular: [N, 3(RGB)]
        :param shininess: [N] or int
        """
        direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
        self.lights = DirectionalLights(ambient_color=ambient, diffuse_color=diffuse, specular_color=specular,
                                        direction=direction, device=self.device)

        self.materials = Materials(shininess=shininess, device=self.device)
        # multiple colors are not supported yet
        # pytorch3d bug: material batch dimension will be broadcast to color faces (k) dimension
        # pytorch3d/renderer/mesh/shading.py: _apply_lighting
        self.materials.ambient_color = self.materials.ambient_color[:1]
        self.materials.diffuse_color = self.materials.diffuse_color[:1]
        self.materials.specular_color = self.materials.specular_color[:1]

    def apply_lighting(self, pixel_coords: torch.Tensor, pixel_normals: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param pixel_coords: [N, H, W, K, 3(XYZ)]
        :param pixel_normals: [N, H, W, K, 3(XYZ)]
        :param texels: [N, H, W, K, 3(RGB)]
        :param ambient: [N, 3(RGB)]
        :param diffuse: [N, 3(RGB)]
        :param specular: [N, 3(RGB)]
        :param direction: [N, 3(XYZ)], un-normalized
        :param shininess: [N] or int
        :return: [N, H, W, K, 3(RGB)]
        """
        ambient, diffuse, specular = \
            _apply_lighting(pixel_coords, pixel_normals, self.lights, self.cameras, self.materials)
        return ambient * pixel_normals.bool().any(dim=-1)[..., None] + diffuse, specular
