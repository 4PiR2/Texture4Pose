import torch
from torch import nn
from pytorch3d.renderer import TexturesVertex

from dataloader.obj_mesh import ObjMesh
import utils.transform_3d


class TextureNetV(nn.Module):
    def __init__(self, objects: dict[int, ObjMesh]):
        super().__init__()
        weights = {}
        for oid in objects:
            obj = objects[oid]
            verts = obj.mesh.verts_packed()  # [V, 3(XYZ)]
            # texture = torch.zeros_like(verts)
            # texture = torch.full_like(verts, .5)
            texture = utils.transform_3d.normalize_coord_3d(verts.reshape(3, -1, 1), obj.size).reshape(-1, 3)
            weights[str(oid)] = nn.Parameter(texture)
        self.weights = nn.ParameterDict(weights)
        self.act = nn.Sigmoid()

    def forward(self, obj: ObjMesh):
        texture = self.weights[str(obj.obj_id)]
        # texture = self.act(texture)
        return TexturesVertex(texture.to(torch.float32)[None])
