from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import HardPhongShader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
import torch
from torch.nn import functional as F

from config import const as cc


class GTHardPhongShader(HardPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """
    def __init__(self, dtype: torch.dtype = cc.dtype, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        verts = meshes.verts_packed()  # [V, 3(XYZ)]
        faces = meshes.faces_packed()  # [F, 3]
        vertex_normals = meshes.verts_normals_packed()  # [V, 3(XYZ)]

        fragments_zbuf, fragments_bary_coords, fragments_dists, verts, vertex_normals = [x.to(dtype=self.dtype)
            for x in [fragments.zbuf, fragments.bary_coords, fragments.dists, verts, vertex_normals]]
        fragments = Fragments(fragments.pix_to_face, fragments_zbuf, fragments_bary_coords, fragments.dists)

        faces_verts = verts[faces]  # [F, 3, 3(XYZ)]
        faces_normals = vertex_normals[faces]  # [F, 3, 3(XYZ)]
        pixel_coords = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts)
        # [N, H, W, K, C]
        pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals)
        # [N, H, W, K, C]
        pixel_normals = F.normalize(pixel_normals, p=2, dim=-1, eps=1e-6)

        # pix_to_face = fragments.pix_to_face  # [N, H, W, K]
        zbuf = torch.where(fragments.zbuf >= 0., fragments.zbuf,
                           torch.tensor(torch.inf, dtype=self.dtype, device=fragments.zbuf.device))  # [N, H, W, K]

        if meshes.textures is not None:
            texels = meshes.sample_textures(fragments)
        else:
            texels = None
        return pixel_coords, pixel_normals, zbuf, texels
