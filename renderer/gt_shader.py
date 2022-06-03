from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import HardPhongShader, Materials
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shading import _apply_lighting
from pytorch3d.structures import Meshes
import torch


def gt_phong_shading(
    meshes: Meshes, fragments: Fragments, lights, cameras: CamerasBase, materials: Materials, texels: torch.Tensor
):
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # [V, 3(XYZ)]
    faces = meshes.faces_packed()  # [F, 3]
    vertex_normals = meshes.verts_normals_packed()  # [V, 3(XYZ)]
    faces_verts = verts[faces]  # [F, 3, 3(XYZ)]
    faces_normals = vertex_normals[faces]  # [F, 3, 3(XYZ)]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )  # [N, H, W, K, C]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )  # [N, H, W, K, C]
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular  # [N, H, W, K, C]
    return colors, pixel_coords, pixel_normals


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
    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        # blend_params = kwargs.get("blend_params", self.blend_params)
        colors, pixel_coords, pixel_normals = gt_phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # images = hard_rgb_blend(colors, fragments, blend_params)
        texels = texels[..., 0, :].permute(0, 3, 1, 2)  # [N, 3(RGB), H, W]
        colors = colors[..., 0, :].permute(0, 3, 1, 2)  # [N, 3(RGB), H, W]
        # pix_to_face = fragments.pix_to_face[:, None, ..., 0]  # [N, 1, H, W], fragments.pix_to_face: [N, H, W, K]
        pixel_coords = pixel_coords[..., 0, :].permute(0, 3, 1, 2)  # [N, 3(XYZ), H, W]
        pixel_normals = pixel_normals[..., 0, :].permute(0, 3, 1, 2)  # [N, 3(XYZ), H, W]
        zbuf = fragments.zbuf[:, None, ..., 0]  # [N, 1, H, W]
        mask_obj = zbuf >= 0.
        zbuf[~mask_obj] = torch.inf
        mask_vis = (zbuf == zbuf.min(dim=0)[0]) & mask_obj  # [N, 1, H, W]
        return colors, texels, mask_obj, mask_vis, pixel_coords, pixel_normals
