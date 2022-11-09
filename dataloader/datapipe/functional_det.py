import torch
from torch.utils.data import functional_datapipe

from dataloader.datapipe.helper import SampleMapperIDP
from dataloader.sample import SampleFields as sf


@functional_datapipe('calibrate_bbox_x1x2y1y2_abs')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP):
        super().__init__(src_dp, [sf.cam_K, sf.gt_bbox_vis], [sf.bbox])

    def main(self, cam_K: torch.Tensor, gt_bbox_vis: torch.Tensor) -> torch.Tensor:
        x, y, w, h = gt_bbox_vis.split(1, dim=-1)
        x1 = x - w * .5
        x2 = x + w * .5
        y1 = y - h * .5
        y2 = y + h * .5
        x1y1 = cam_K[..., :-1, :] @ torch.cat([x1, y1, torch.ones_like(x1)], dim=-1)[..., None]
        x2y2 = cam_K[..., :-1, :] @ torch.cat([x2, y2, torch.ones_like(x1)], dim=-1)[..., None]
        bbox = torch.cat([x1y1[..., 0], x2y2[..., 0]], dim=-1)
        return bbox
