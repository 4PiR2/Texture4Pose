import torch
from torch import nn
import torch.nn.functional as F

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample, SampleFields as sf
import utils.image_2d
import utils.transform_3d


class Loss(nn.Module):
    def __init__(self, objects_eval: dict[int, ObjMesh], pose: bool = True, geo: bool = True, mean: bool = True):
        super().__init__()
        self.objects_eval: dict[int, ObjMesh] = objects_eval
        self.pose: bool = pose
        self.geo: bool = geo
        self.mean: bool = mean

    def forward(self, sample: Sample, pred_cam_R_m2c: torch.Tensor = None, pred_cam_t_m2c_site: torch.Tensor = None,
                pred_coord_3d_roi: torch.Tensor = None, pred_mask_vis_roi: torch.Tensor = None):
        losses = []
        if self.pose:
            obj_id = sample.get(sf.obj_id)
            gt_cam_R_m2c = sample.get(sf.gt_cam_R_m2c)
            gt_cam_t_m2c_site = sample.get(sf.gt_cam_t_m2c_site)

            if pred_cam_R_m2c is None:
                pred_cam_R_m2c = sample.get(sf.pred_cam_R_m2c)
            if pred_cam_t_m2c_site is None:
                pred_cam_t_m2c_site = sample.get(sf.pred_cam_t_m2c_site)

            pm_loss = self.pm_loss(obj_id, gt_cam_R_m2c, pred_cam_R_m2c, div_diameter=True)
            t_site_center_loss = self.t_site_center_loss(gt_cam_t_m2c_site, pred_cam_t_m2c_site)
            t_site_depth_loss = self.t_site_depth_loss(gt_cam_t_m2c_site, pred_cam_t_m2c_site)
            losses += [pm_loss, t_site_center_loss, t_site_depth_loss]

        if self.geo:
            gt_coord_3d_roi_normalized = sample.get(sf.gt_coord_3d_roi_normalized)
            gt_mask_vis_roi = sample.get(sf.gt_mask_vis_roi)

            if pred_coord_3d_roi is None:
                pred_coord_3d_roi_normalized = sample.get(sf.pred_coord_3d_roi_normalized)
            else:
                obj_size = sample.get(sf.obj_size)
                pred_coord_3d_roi_normalized = utils.transform_3d.normalize_coord_3d(pred_coord_3d_roi, obj_size)
            if pred_mask_vis_roi is None:
                pred_mask_vis_roi = sample.get(sf.pred_mask_vis_roi)

            coord_3d_loss = self.coord_3d_loss(gt_coord_3d_roi_normalized, gt_mask_vis_roi, pred_coord_3d_roi_normalized)
            mask_loss = self.mask_loss(gt_mask_vis_roi, pred_mask_vis_roi)
            losses += [coord_3d_loss, mask_loss]
        if self.mean:
            losses = [loss.mean() for loss in losses]
        return losses

    def pm_loss(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, pred_cam_R_m2c: torch.Tensor = None,
                div_diameter: bool = True) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param div_diameter: bool
        :return: [N]
        """
        N = len(obj_id)
        loss = torch.empty(N, dtype=pred_cam_R_m2c.dtype, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = self.objects_eval[int(obj_id[i])]
            loss[i] = obj.average_distance(pred_cam_R_m2c[i], gt_cam_R_m2c[i], p=1)
            if div_diameter:
                loss[i] /= obj.diameter
        return loss

    def t_site_center_loss(self, gt_cam_t_m2c_site: torch.Tensor, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.linalg.vector_norm(gt_cam_t_m2c_site[:, :2] - pred_cam_t_m2c_site[:, :2], ord=1, dim=-1)

    def t_site_depth_loss(self, gt_cam_t_m2c_site: torch.Tensor, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.abs(gt_cam_t_m2c_site[:, 2] - pred_cam_t_m2c_site[:, 2])

    def coord_3d_loss(self, gt_coord_3d_roi_normalized: torch.Tensor, gt_mask_vis_roi:torch.Tensor,
                      pred_coord_3d_roi_normalized: torch.Tensor) -> torch.Tensor:
        """
        :param pred_coord_3d_roi_normalized: [..., 3(XYZ), H, W]
        :return: [...]
        """
        loss = utils.image_2d.lp_loss(
            (pred_coord_3d_roi_normalized - gt_coord_3d_roi_normalized) * gt_mask_vis_roi, p=1, reduction='sum')
        loss /= gt_mask_vis_roi.sum(dim=[-3, -2, -1])
        return loss

    def mask_loss(self, gt_mask_vis_roi: torch.Tensor, pred_mask_vis_roi: torch.Tensor, loss_mode: str = 'BCE') -> torch.Tensor:
        """
        :param pred_mask_vis_roi: [..., 1, H, W]
        :param loss_mode: 'BCE', 'L1'
        :return: [...]
        """
        if loss_mode == 'BCE':
            loss = F.binary_cross_entropy(pred_mask_vis_roi, gt_mask_vis_roi.to(dtype=pred_mask_vis_roi.dtype),
                                          reduction='none')
            return loss.mean(dim=[-3, -2, -1])
        elif loss_mode == 'L1':
            pred_mask_vis_roi = utils.image_2d.conditional_clamp(pred_mask_vis_roi, gt_mask_vis_roi, l0=0., u1=1.)
            return utils.image_2d.lp_loss(pred_mask_vis_roi, gt_mask_vis_roi.to(dtype=pred_mask_vis_roi.dtype),
                                          p=1)
        else:
            raise NotImplementedError
