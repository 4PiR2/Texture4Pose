import torch
from pytorch3d.renderer import TexturesVertex
from torch import nn

from dataloader.obj_mesh import ObjMesh


class TextureNet(nn.Module):
    def __init__(self, objects: dict[int, ObjMesh]):
        super().__init__()
        weights = {str(oid): nn.Parameter(torch.zeros_like(objects[oid].mesh.verts_packed()))
                   for oid in objects}
        self.weights = nn.ParameterDict(weights)
        self.act = nn.Sigmoid()

    def forward(self, obj: ObjMesh):
        texture = self.act(self.weights[str(obj.obj_id)])
        return TexturesVertex(texture[None])
