import math

import torch
import torch.nn.functional as F
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.mesh import TexturesBase
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from utils.const import dtype
from utils.cube_mesh import cube
from utils.io import parse_device
from utils.mbsc.exact_min_bound_sphere_3D import exact_min_bound_sphere_3D


class ObjMesh:
    def __init__(self, mesh=None, dtype=dtype, device=None, obj_id=None, name=None, is_eval=None, diameter=None,
                 min=None, size=None, radius=None, center=None, **kwargs):
        if not hasattr(self, 'dtype'):
            self.dtype: torch.dtype = dtype
        if not hasattr(self, 'device'):
            self.device: torch.device = parse_device(device)
        if not hasattr(self, 'mesh'):
            self.mesh: Meshes = mesh.to(self.device)
        if not hasattr(self, 'obj_id'):
            self.obj_id: int = int(obj_id) if obj_id is not None else None
        if not hasattr(self, 'name'):
            self.name: str = str(name) if name is not None else None
        if not hasattr(self, 'is_eval'):
            self.is_eval: bool = bool(is_eval) if is_eval is not None else None
        if not hasattr(self, 'diameter'):
            self.diameter: float = float(diameter) if diameter is not None \
                else float(F.pdist(self.mesh.verts_packed()).max())
        if not hasattr(self, 'min'):
            self.min: torch.Tensor = self._cvt_tensor(min) if min is not None \
                else self.mesh.get_bounding_boxes()[0, :, 0]
        if not hasattr(self, 'size'):
            if size is not None:
                self.size: torch.Tensor = self._cvt_tensor(size)
            else:
                bbox3d = self.mesh.get_bounding_boxes()
                self.size: torch.Tensor = bbox3d[0, :, 1] - bbox3d[0, :, 0]
        if not hasattr(self, 'radius') or not hasattr(self, 'center'):
            if radius is not None and center is not None:
                self.radius: float = radius
                self.center: torch.Tensor = self._cvt_tensor(center)
            else:
                radius, center = self._compute_min_sphere()
                self.radius: float = float(radius)
                self.center: torch.Tensor = self._cvt_tensor(center)

    def _cvt_tensor(self, array) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(array, device=self.device, dtype=self.dtype)

    def _compute_min_sphere(self):
        while True:
            try:
                radius, center, _ = exact_min_bound_sphere_3D(self.mesh.verts_packed().detach().cpu().numpy())
            except:
                pass
            else:
                return radius, center

    def get_transformed_mesh(self, cam_R_m2c, cam_t_m2c) -> Meshes:
        verts = self.mesh.verts_packed()
        return self.mesh.offset_verts(verts @ (cam_R_m2c.T - torch.eye(3, device=self.device)) + cam_t_m2c)

    def get_texture(self, f=None) -> TexturesBase:
        if f is None:
            if self.mesh.textures is not None:
                return self.mesh.textures
            else:
                return TexturesVertex(torch.ones_like(self.mesh.verts_packed())[None])
        else:
            return f(self)

    def average_distance(self, R1: torch.Tensor, R2: torch.Tensor, t1: torch.Tensor = None, t2: torch.Tensor = None,
                         p: int = 2) -> torch.Tensor:
        """
        average distance
        http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

        :param R1: [..., 3, 3]
        :param t1: [..., 3]
        :param R2: [..., 3, 3]
        :param t2: [..., 3]
        :param p: int
        :return: [...]
        """
        verts = self.mesh.verts_packed().expand(*R1.shape[:-2], -1, -1)  # [..., V, 3]
        errors = verts @ (R1 - R2).transpose(-2, -1)  # [..., V, 3]
        if t1 is not None and t2 is not None:
            errors += (t1 - t2)[..., None, :]  # dt: [..., 1, 3]
        distances = torch.linalg.vector_norm(errors, ord=p, dim=-1)  # [..., V]
        return distances.mean(dim=-1)  # [...]

    def average_projected_distance(self, cam_K: torch.Tensor, R1: torch.Tensor, R2: torch.Tensor,
                                   t1: torch.Tensor = None, t2: torch.Tensor = None, p: int = 2) -> torch.Tensor:
        """
        average projected distance (in pixel)
        https://github.com/ybkscht/EfficientPose/blob/main/eval/common.py

        :param cam_K: [..., 3, 3]
        :param R1: [..., 3, 3]
        :param t1: [..., 3]
        :param R2: [..., 3, 3]
        :param t2: [..., 3]
        :param p: int
        :return: [...]
        """
        if cam_K is None:
            cam_K = torch.eye(3, dtype=R1.dtype, device=R1.device)
        if t1 is None:
            t1 = torch.zeros_like(R1[..., 0])
        if t2 is None:
            t2 = torch.zeros_like(R1[..., 0])
        P1 = cam_K @ torch.cat([R1, t1[..., None]], dim=-1)  # [..., 3, 4]
        P2 = cam_K @ torch.cat([R2, t2[..., None]], dim=-1)  # [..., 3, 4]
        verts = self.mesh.verts_packed().expand(*R1.shape[:-2], -1, -1)  # [..., V, 3]
        verts = torch.cat([verts, torch.ones_like(verts[..., -1:])], dim=-1)  # [..., V, 4]
        pv1 = verts @ P1.transpose(-2, -1)  # [..., V, 3]
        pv2 = verts @ P2.transpose(-2, -1)  # [..., V, 3]
        pv1 = pv1[..., :2] / pv1[..., 2:]  # [..., V, 2]
        pv2 = pv2[..., :2] / pv2[..., 2:]  # [..., V, 2]
        distances = torch.linalg.vector_norm(pv1 - pv2, ord=p, dim=-1)  # [..., V]
        return distances.mean(dim=-1)  # [...]


class BOPMesh(ObjMesh):
    scale: float = 1e-3  # prevent overflow in PyTorch3D, 1e-3 means using meter as unit length

    def __init__(self, mesh_path=None, diameter=None, min_x=None, min_y=None, min_z=None, size_x=None, size_y=None,
                 size_z=None, symmetries_continuous=None, symmetries_discrete=None, **kwargs):
        self.symmetries_continuous = symmetries_continuous
        self.symmetries_discrete = symmetries_discrete
        if diameter is not None:
            diameter *= BOPMesh.scale
        min = torch.tensor((min_x, min_y, min_z)) * BOPMesh.scale if min_x is not None else None
        size = torch.tensor((size_x, size_y, size_z)) * BOPMesh.scale if size_x is not None else None
        mesh = IO().load_mesh(mesh_path).scale_verts_(BOPMesh.scale)
        super().__init__(mesh=mesh, diameter=diameter, min=min, size=size, **kwargs)


class RegularMesh(ObjMesh):
    def __init__(self, name=None, scale=5e-2, **kwargs):
        min = torch.full([3], -scale)
        size = torch.full([3], scale * 2.)
        center = torch.zeros(3)
        if name == 'sphere':
            radius = scale
            mesh = ico_sphere(level=1).scale_verts_(scale)
        elif name == 'cube':
            radius = scale * math.sqrt(2.)
            mesh = cube(level=1).scale_verts_(scale)
        else:
            raise NotImplementedError
        diameter = radius * 2.
        super().__init__(mesh=mesh, name=name, diameter=diameter, min=min, size=size, radius=radius, center=center,
                         **kwargs)
