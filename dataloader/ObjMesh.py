import torch
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from utils.io import parse_device
from utils.mbsc.exact_min_bound_sphere_3D import exact_min_bound_sphere_3D
from utils.const import debug_mode


class ObjMesh:
    scale: float = 1e-3  # prevent overflow in PyTorch3D, 1e-3 means using meter as unit length

    def __init__(self, device=None, obj_id=None, name=None, is_eval=None, mesh_path=None, diameter=None,
                 min_x=None, min_y=None, min_z=None,
                 size_x=None, size_y=None, size_z=None,
                 symmetries_continuous=None, symmetries_discrete=None):
        self.device: torch.device = parse_device(device)
        self.obj_id: int = obj_id
        self.name: str = name
        self.is_eval: bool = is_eval
        self.mesh_path: str = mesh_path
        self.diameter: float = diameter * ObjMesh.scale
        self.min: torch.Tensor = torch.tensor((min_x, min_y, min_z), device=self.device) * ObjMesh.scale
        self.size: torch.Tensor = torch.tensor((size_x, size_y, size_z), device=self.device) * ObjMesh.scale
        self.symmetries_continuous = symmetries_continuous
        self.symmetries_discrete = symmetries_discrete
        self.mesh: Meshes = IO().load_mesh(self.mesh_path).to(self.device).scale_verts_(ObjMesh.scale)
        radius, center, _ = exact_min_bound_sphere_3D(self.mesh.verts_packed().cpu().numpy())
        self.radius: float = radius
        self.center: torch.Tensor = torch.tensor(center).to(self.device)

    def get_transformed_mesh(self, cam_R_m2c, cam_t_m2c) -> Meshes:
        verts = self.mesh.verts_packed()
        return self.mesh.offset_verts(verts @ (cam_R_m2c.T - torch.eye(3, device=self.device)) + cam_t_m2c)

    def get_gt_texture(self):
        verts = self.mesh.verts_packed()  # [V, 3(XYZ)]
        gt_texture = torch.cat([torch.full((len(verts), 1), self.obj_id, device=self.device), verts], dim=1)
        return TexturesVertex(gt_texture[None])  # [1, V, 4(IXYZ)]

    def get_texture(self, f=None):
        if f is None:
            return self.mesh.textures
        else:
            return f(self)

    def point_match_error(self, R1: torch.Tensor, R2: torch.Tensor, p: int = 1) -> torch.Tensor:
        """
        :param R1: [B, 3, 3]
        :param R2: [B, 3, 3]
        :param p: int
        :return: [B]
        """
        if R1.dim() < 3:
            R1 = R1[None]
        if R2.dim() < 3:
            R2 = R2[None]
        errors = torch.norm(self.mesh.verts_packed()[None] @ (R1 - R2).transpose(-2, -1), p=p, dim=-1)
        return errors.mean(dim=-1)
