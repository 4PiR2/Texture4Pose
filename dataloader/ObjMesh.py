import torch
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.mesh import TexturesBase
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from utils.const import dtype
from utils.io import parse_device
from utils.mbsc.exact_min_bound_sphere_3D import exact_min_bound_sphere_3D


class BOPMesh:
    scale: float = 1e-3  # prevent overflow in PyTorch3D, 1e-3 means using meter as unit length

    def __init__(self, dtype=dtype, device=None, obj_id=None, name=None, is_eval=None, mesh_path=None, diameter=None,
                 min_x=None, min_y=None, min_z=None,
                 size_x=None, size_y=None, size_z=None,
                 symmetries_continuous=None, symmetries_discrete=None):
        self.dtype: torch.dtype = dtype
        self.device: torch.device = parse_device(device)
        self.obj_id: int = obj_id
        self.name: str = name
        self.is_eval: bool = is_eval
        self.mesh_path: str = mesh_path
        self.symmetries_continuous = symmetries_continuous
        self.symmetries_discrete = symmetries_discrete
        self.diameter: float = diameter * BOPMesh.scale if diameter is not None else None
        self.min: torch.Tensor = torch.tensor((min_x, min_y, min_z), dtype=self.dtype, device=self.device) * \
                                 BOPMesh.scale if min_x is not None else None
        self.size: torch.Tensor = torch.tensor((size_x, size_y, size_z), dtype=self.dtype, device=self.device) * \
                                  BOPMesh.scale if size_x is not None else None

        if self.mesh_path is not None:
            self.mesh: Meshes = IO().load_mesh(self.mesh_path, device=self.device).scale_verts_(BOPMesh.scale)
            radius, center, _ = exact_min_bound_sphere_3D(self.mesh.verts_packed().cpu().numpy())
            self.radius: float = radius
            self.center: torch.Tensor = torch.tensor(center, device=self.device, dtype=self.dtype)

    def get_transformed_mesh(self, cam_R_m2c, cam_t_m2c) -> Meshes:
        verts = self.mesh.verts_packed()
        return self.mesh.offset_verts(verts @ (cam_R_m2c.T - torch.eye(3, device=self.device)) + cam_t_m2c)

    def get_texture(self, f=None) -> TexturesBase:
        if f is None:
            return self.mesh.textures
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
        distances = torch.norm(errors, p=p, dim=-1)  # [..., V]
        return distances.mean(dim=-1)  # [...]

class RegularMesh(BOPMesh):
    scale: float = 100e-3

    def __init__(self, dtype=dtype, device=None, obj_id=None, name=None):
        super().__init__(dtype, device, obj_id, name)
        self.diameter: float = BOPMesh.scale * 2.
        self.min: torch.Tensor = torch.full([3], -BOPMesh.scale, dtype=self.dtype, device=self.device)
        self.size: torch.Tensor = torch.full([3], BOPMesh.scale * 2., dtype=self.dtype, device=self.device)
        self.mesh: Meshes = ico_sphere(level=1, device=self.device).scale_verts_(BOPMesh.scale)
        self.mesh.textures = TexturesVertex(torch.ones_like(self.mesh.verts_packed()[None]))
        self.radius: float = BOPMesh.scale
        self.center: torch.Tensor = torch.zeros(3, device=self.device, dtype=self.dtype)
