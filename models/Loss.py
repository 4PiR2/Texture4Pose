from torch import nn

from dataloader.PoseDataset import BOPObjDataset
from dataloader.Sample import Sample


class Loss(nn.Module):
    def __init__(self, dataset: BOPObjDataset):
        super().__init__()
        self.dataset = dataset
        self.device = self.dataset.device

    def forward(self, sample: Sample, pred_cam_R_m2c, pred_cam_t_m2c_site):
        # y = (x @ R.T + t) @ K.T
        t_site_loss = sample.t_site_center_loss(pred_cam_t_m2c_site) + sample.t_site_depth_loss(pred_cam_t_m2c_site)
        pm_loss = sample.pm_loss(self.dataset.objects_eval, pred_cam_R_m2c)
        loss = t_site_loss.sum() + pm_loss.sum()
        return loss
