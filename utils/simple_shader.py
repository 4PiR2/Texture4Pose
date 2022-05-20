import torch
import torch.nn as nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


class SimpleShader(nn.Module):
    def __init__(self, background: float = 0.):
        super().__init__()
        self.background = background

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = BlendParams(background_color=torch.full((texels.shape[-1],), self.background, dtype=texels.dtype,
                                                               device=texels.device))
        images = hard_rgb_blend(texels, fragments, blend_params)  # (N, H, W, D+1) RGBA image
        return images[..., :-1]  # (N, H, W, D)
