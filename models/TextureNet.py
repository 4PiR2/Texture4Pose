import torch
from torch import nn

from dataloader.ObjMesh import BOPMesh


class TextureNet(nn.Module):
    def __init__(self, objects: dict[int, BOPMesh]):
        super().__init__()
        weights = {oid: nn.Parameter(torch.zeros_like(objects[oid].mesh.verts_packed()))
                   for oid in objects}
        self.weights = nn.ParameterDict(weights)
        self.act = nn.Sigmoid()

    def forward(self, obj: BOPMesh):
        return self.act(self.weights[obj.obj_id])
