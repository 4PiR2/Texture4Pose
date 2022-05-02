import torch
from pytorch3d.transforms import so3_relative_angle, rotation_6d_to_matrix

from dataloader.BOPDataset import BOPDataset
from dataloader.Sample import Sample
from utils.const import pnp_input_size
from utils.transform import calculate_bbox_crop, t_site_to_t


class Loss:
    def __init__(self, dataset: BOPDataset, sample: Sample):
        self.dataset = dataset
        self.sample = sample
        self.device = self.dataset.device

    def eval_pred(self, pred_cam_R_m2c_6d, pred_cam_t_m2c_site):
        # y = (x @ R.T + t) @ K.T
        pred_cam_R_m2c = rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        crop_size, *_ = calculate_bbox_crop(self.sample.bbox)
        pred_cam_t_m2c = t_site_to_t(pred_cam_t_m2c_site, self.sample.bbox,
                                     pnp_input_size / crop_size, self.sample.cam_K)

        t_site_loss = self.t_site_center_loss(pred_cam_t_m2c_site) + self.t_site_depth_loss(pred_cam_t_m2c_site)
        pm_loss = self.pm_loss(pred_cam_R_m2c)
        loss = t_site_loss.sum() + pm_loss.sum()

        angle = self.relative_angle(pred_cam_R_m2c)
        dist = self.relative_dist(pred_cam_t_m2c)
        return loss, angle, dist

    def pm_loss(self, pred_cam_R_m2c: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_R_m2c: [N, 3, 3]
        :return: [N]
        """
        n = len(self.sample.obj_id)
        loss = torch.empty(n, device=self.device)
        for i in range(n):
            obj_id = int(self.sample.obj_id[i])
            loss[i] = self.dataset.objects_eval[obj_id].point_match_error(pred_cam_R_m2c[i], self.sample.gt_cam_R_m2c[i])
        return loss

    def t_site_center_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.norm(self.sample.gt_cam_t_m2c_site[:, :2] - pred_cam_t_m2c_site[:, :2], p=1, dim=-1)

    def t_site_depth_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """

        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.norm(self.sample.gt_cam_t_m2c_site[:, 2] - pred_cam_t_m2c_site[:, 2], p=1, dim=-1)

    def relative_angle(self, pred_cam_R_m2c: torch.Tensor) -> torch.Tensor:
        return so3_relative_angle(pred_cam_R_m2c, self.sample.gt_cam_R_m2c, eps=1e-2)

    def relative_dist(self, pred_cam_t_m2c: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred_cam_t_m2c - self.sample.gt_cam_t_m2c, p=2, dim=-1)
