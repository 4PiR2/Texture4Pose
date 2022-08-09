import pytorch3d.transforms
import torch
import torch.nn.functional as F
from torch.utils.data import functional_datapipe

from dataloader.datapipe.helper import SampleMapperIDP
from dataloader.sample import SampleFields as sf


@functional_datapipe('rand_gt_rotation_cylinder')
class _(SampleMapperIDP):
    def __init__(self, src_dp: SampleMapperIDP, thresh_theta: float = 15. * torch.pi / 180.):
        super().__init__(src_dp, [sf.gt_cam_t_m2c], [sf.gt_cam_R_m2c], required_attributes=['dtype', 'device'])
        self._cos_thresh = torch.tensor(thresh_theta).cos()

    def main(self, gt_cam_t_m2c: torch.Tensor):
        # for cylinderside, empty projection in z-axis direction
        gt_cam_R_m2c = []
        for t in F.normalize(gt_cam_t_m2c, p=2, dim=-1):
            while True:
                rot = pytorch3d.transforms.random_rotations(1, dtype=self.dtype, device=self.device)[0]
                if torch.dot(t, rot[:, 2]).abs() <= self._cos_thresh:
                    gt_cam_R_m2c.append(rot)
                    break
        return torch.stack(gt_cam_R_m2c, dim=0)
