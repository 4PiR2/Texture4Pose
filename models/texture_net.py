import torch
from torch import nn
from pytorch3d.renderer import TexturesVertex

from dataloader.obj_mesh import ObjMesh


class TextureNet(nn.Module):
    def __init__(self, objects: dict[int, ObjMesh]):
        super().__init__()
        weights = {str(oid): nn.Parameter(torch.full_like(objects[oid].mesh.verts_packed(), 0.))
                   for oid in objects}
        self.weights = nn.ParameterDict(weights)
        self.act = nn.Sigmoid()

    def forward(self, obj: ObjMesh):
        texture = self.weights[str(obj.obj_id)]
        texture = self.act(texture)
        return TexturesVertex(texture[None])
