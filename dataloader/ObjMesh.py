import torch
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex

from utils.mbsc.exact_min_bound_sphere_3D import exact_min_bound_sphere_3D
from utils.const import debug_mode


class ObjMesh:
    scale = 1e-3  # prevent overflow in PyTorch3D, 1e-3 means using meter as unit length

    def __init__(self, device=None, obj_id=None, name=None, is_eval=None, mesh_path=None, diameter=None,
                 min_x=None, min_y=None, min_z=None,
                 size_x=None, size_y=None, size_z=None,
                 symmetries_continuous=None, symmetries_discrete=None):
        self.device = device if device is not None else 'cpu'
        self.obj_id = obj_id
        self.name = name
        self.is_eval = is_eval
        self.mesh_path = mesh_path
        self.diameter = diameter * ObjMesh.scale
        self.min = torch.tensor((min_x, min_y, min_z), device=self.device) * ObjMesh.scale
        self.size = torch.tensor((size_x, size_y, size_z), device=self.device) * ObjMesh.scale
        self.symmetries_continuous = symmetries_continuous
        self.symmetries_discrete = symmetries_discrete
        self.mesh = IO().load_mesh(self.mesh_path).to(self.device).scale_verts_(ObjMesh.scale)
        radius, center, _ = exact_min_bound_sphere_3D(self.mesh.verts_packed().cpu().numpy())
        self.radius, self.center = radius, torch.tensor(center).to(self.device)

    def get_transformed_mesh(self, cam_R_m2c, cam_t_m2c):
        verts = self.mesh.verts_packed()
        return self.mesh.offset_verts(verts @ (cam_R_m2c.T - torch.eye(3, device=self.device)) + cam_t_m2c)

    def get_gt_texture(self, debug=debug_mode):
        verts = self.mesh.verts_packed()  # [V, 3(XYZ)]
        if debug:
            verts_min, verts_max = verts.min(), verts.max()  # scalar
            verts = ((verts - verts_min) * (1 / (verts_max - verts_min))).flip(-1)  # [V, 3(RGB)] \in [0., 1.]
        gt_texture = torch.cat([torch.full((len(verts), 1), self.obj_id, device=self.device), verts], dim=1)
        return TexturesVertex(gt_texture[None])  # [1, V, 4(IXYZ)]

    def get_texture(self, f=None):
        if f is None:
            return self.mesh.textures
        else:
            return f(self)
