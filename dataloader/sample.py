import copy
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dataloader.obj_mesh import ObjMesh
import utils.image_2d
import utils.transform_3d


def _get_param(*elems):
    for elem in elems:
        if isinstance(elem, tuple) and isinstance(elem[1], Callable):
            if elem[0] is not None:
                elem = elem[1](elem[0])
            else:
                elem = None
        elif isinstance(elem, Callable):
            elem = elem()
        if elem is not None:
            return elem
    return None


class Sample:
    def __init__(self, obj_id=None, obj_size=None, cam_K=None, gt_cam_R_m2c=None, gt_cam_t_m2c=None,
                 coord_2d_roi=None, gt_coord_3d_roi=None, gt_mask_vis_roi=None,
                 gt_mask_obj_roi=None, img_roi=None, dbg_img=None, bbox=None, gt_bbox_vis=None, gt_bbox_obj=None,
                 bbox_zoom_out_ratio=None):
        self.obj_id: torch.Tensor = obj_id
        self.obj_size: torch.Tensor = obj_size
        self.cam_K: torch.Tensor = cam_K
        self.gt_cam_R_m2c: torch.Tensor = gt_cam_R_m2c
        self.gt_cam_t_m2c: torch.Tensor = gt_cam_t_m2c
        self.coord_2d_roi: torch.Tensor = coord_2d_roi
        self.gt_coord_3d_roi: torch.Tensor = gt_coord_3d_roi
        self.gt_mask_vis_roi: torch.Tensor = gt_mask_vis_roi
        self.gt_mask_obj_roi: torch.Tensor = gt_mask_obj_roi
        self.img_roi: torch.Tensor = img_roi
        self.dbg_img: torch.Tensor = dbg_img
        self.bbox: torch.Tensor = bbox
        self.gt_bbox_vis: torch.Tensor = gt_bbox_vis
        self.gt_bbox_obj: torch.Tensor = gt_bbox_obj
        self.bbox_zoom_out_ratio: float = bbox_zoom_out_ratio

        self.pred_cam_R_m2c: Optional[torch.Tensor] = None
        self.pred_cam_t_m2c: Optional[torch.Tensor] = None
        self.pred_cam_t_m2c_site: Optional[torch.Tensor] = None
        self.pred_coord_3d_roi: Optional[torch.Tensor] = None
        self.pred_coord_3d_roi_normalized: Optional[torch.Tensor] = None
        self.pred_mask_vis_roi: Optional[torch.Tensor] = None
        self.pnp_cam_R_m2c: Optional[torch.Tensor] = None
        self.pnp_cam_t_m2c: Optional[torch.Tensor] = None
        self.pnp_inlier_roi: Optional[torch.Tensor] = None

    @classmethod
    def collate(cls, batch: list):
        keys = [key for key in dir(batch[0]) if not key.startswith('__') and not callable(getattr(batch[0], key))]
        out = cls()
        for key in keys:
            if not isinstance(getattr(batch[0], key), torch.Tensor) or key in []:
                setattr(out, key, getattr(batch[0], key))
            else:
                setattr(out, key, torch.cat([getattr(b, key) for b in batch], dim=0))
        return out

    def compute_pnp(self, sanity_check_mode: bool = False, store: bool = True, ransac: bool = True,
                erode_min: float = 0., erode_max: float = torch.inf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sanity_check_mode:
            pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi = utils.transform_3d.solve_pnp(self.gt_coord_3d_roi,
                self.coord_2d_roi, utils.image_2d.erode_mask(self.gt_mask_vis_roi, erode_min, erode_max), ransac)
        else:
            pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi = utils.transform_3d.solve_pnp(self.pred_coord_3d_roi,
                self.coord_2d_roi, utils.image_2d.erode_mask(self.pred_mask_vis_roi, erode_min, erode_max), ransac)
        if store:
            self.pnp_cam_R_m2c, self.pnp_cam_t_m2c, self.pnp_inlier_roi \
                = pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi
        return pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi

    def _get_roi_zoom_in_ratio(self) -> torch.Tensor:
        pnp_input_size = max(self.gt_coord_3d_roi.shape[-2:])
        crop_size = utils.image_2d.get_dzi_crop_size(self.bbox, self.bbox_zoom_out_ratio)
        return pnp_input_size / crop_size  # [N]

    def _get_cam_t_m2c_site(self, cam_t_m2c: torch.Tensor) -> torch.Tensor:
        return utils.transform_3d.t_to_t_site(cam_t_m2c, self.bbox, self._get_roi_zoom_in_ratio(), self.cam_K)  # [N, 3]

    def _get_cam_t_m2c(self, cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        return utils.transform_3d.t_site_to_t(cam_t_m2c_site, self.bbox, self._get_roi_zoom_in_ratio(), self.cam_K)
        # [N, 3]

    def gt_cam_t_m2c_site(self) -> torch.Tensor:
        return self._get_cam_t_m2c_site(self.gt_cam_t_m2c)  # [N, 3]

    def get_pred_cam_t_m2c(self, store: bool = True) -> torch.Tensor:
        pred_cam_t_m2c = self._get_cam_t_m2c(self.pred_cam_t_m2c_site)
        if store:
            self.pred_cam_t_m2c = pred_cam_t_m2c
        return pred_cam_t_m2c  # [N, 3]

    def _get_coord_3d_roi_normalized(self, coord_3d_roi: torch.Tensor) -> torch.Tensor:
        return utils.transform_3d.normalize_coord_3d(coord_3d_roi, self.obj_size)  # [N, 3(XYZ), H, W]

    def _get_coord_3d_roi(self, coord_3d_roi_normalized: torch.Tensor) -> torch.Tensor:
        return utils.transform_3d.denormalize_coord_3d(coord_3d_roi_normalized, self.obj_size)  # [N, 3(XYZ), H, W]

    def get_pred_coord_3d_roi(self, store: bool = True) -> torch.Tensor:
        pred_coord_3d_roi = self._get_coord_3d_roi(self.pred_coord_3d_roi_normalized)
        if store:
            self.pred_coord_3d_roi = pred_coord_3d_roi
        return pred_coord_3d_roi  # [N, 3(XYZ), H, W]

    def pm_loss(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor = None, div_diameter: bool = True
                ) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param div_diameter: bool
        :return: [N]
        """
        pred_cam_R_m2c = _get_param(pred_cam_R_m2c, self.pred_cam_R_m2c)
        N = len(self.obj_id)
        loss = torch.empty(N, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = objects_eval[int(self.obj_id[i])]
            loss[i] = obj.average_distance(pred_cam_R_m2c[i], self.gt_cam_R_m2c[i], p=1)
            if div_diameter:
                loss[i] /= obj.diameter
        return loss

    def add_score(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor = None,
                  pred_cam_t_m2c: torch.Tensor = None, div_diameter: bool = True) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :param div_diameter: bool
        :return: [N]
        """
        pred_cam_R_m2c = _get_param(pred_cam_R_m2c, self.pred_cam_R_m2c)
        pred_cam_t_m2c = _get_param(pred_cam_t_m2c, self.pred_cam_t_m2c)
        N = len(self.obj_id)
        add = torch.empty(N, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = objects_eval[int(self.obj_id[i])]
            add[i] = obj.average_distance(pred_cam_R_m2c[i], self.gt_cam_R_m2c[i], pred_cam_t_m2c[i],
                                          self.gt_cam_t_m2c[i])
            if div_diameter:
                add[i] /= obj.diameter
        return add

    def proj_dist(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor = None,
                  pred_cam_t_m2c: torch.Tensor = None) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :return: [N]
        """
        pred_cam_R_m2c = _get_param(pred_cam_R_m2c, self.pred_cam_R_m2c)
        pred_cam_t_m2c = _get_param(pred_cam_t_m2c, self.pred_cam_t_m2c)
        N = len(self.obj_id)
        dist = torch.empty(N, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = objects_eval[int(self.obj_id[i])]
            dist[i] = obj.average_projected_distance(self.cam_K[i], pred_cam_R_m2c[i], self.gt_cam_R_m2c[i],
                                                     pred_cam_t_m2c[i], self.gt_cam_t_m2c[i])
        return dist

    def t_site_center_loss(self, pred_cam_t_m2c_site: torch.Tensor = None) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        pred_cam_t_m2c_site = _get_param(pred_cam_t_m2c_site, self.pred_cam_t_m2c_site)
        return torch.linalg.vector_norm(self.gt_cam_t_m2c_site()[:, :2] - pred_cam_t_m2c_site[:, :2], ord=1, dim=-1)

    def t_site_depth_loss(self, pred_cam_t_m2c_site: torch.Tensor = None) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        pred_cam_t_m2c_site = _get_param(pred_cam_t_m2c_site, self.pred_cam_t_m2c_site)
        return torch.abs(self.gt_cam_t_m2c_site()[:, 2] - pred_cam_t_m2c_site[:, 2])

    def coord_3d_loss(self, pred_coord_3d_roi_normalized: torch.Tensor = None) -> torch.Tensor:
        """
        :param pred_coord_3d_roi_normalized: [..., 3(XYZ), H, W]
        :return: [...]
        """
        pred_coord_3d_roi_normalized = _get_param(pred_coord_3d_roi_normalized, self.pred_coord_3d_roi_normalized,
                                                  lambda: self._get_coord_3d_roi_normalized(self.pred_coord_3d_roi))
        gt_coord_3d_roi_normalized = utils.transform_3d.normalize_coord_3d(self.gt_coord_3d_roi, self.obj_size)
        loss = utils.image_2d.lp_loss(
            (pred_coord_3d_roi_normalized - gt_coord_3d_roi_normalized) * self.gt_mask_vis_roi, p=1, reduction='sum')
        loss /= self.gt_mask_vis_roi.sum(dim=[-3, -2, -1])
        return loss

    def mask_loss(self, pred_mask_vis_roi: torch.Tensor = None, loss_mode: str = 'BCE') -> torch.Tensor:
        """
        :param pred_mask_vis_roi: [..., 1, H, W]
        :param loss_mode: 'BCE', 'L1'
        :return: [...]
        """
        pred_mask_vis_roi = _get_param(pred_mask_vis_roi, self.pred_mask_vis_roi)
        if loss_mode == 'BCE':
            loss = F.binary_cross_entropy(pred_mask_vis_roi, self.gt_mask_vis_roi.to(dtype=pred_mask_vis_roi.dtype),
                                          reduction='none')
            return loss.mean(dim=[-3, -2, -1])
        elif loss_mode == 'L1':
            pred_mask_vis_roi = utils.image_2d.conditional_clamp(pred_mask_vis_roi, self.gt_mask_vis_roi, l0=0., u1=1.)
            return utils.image_2d.lp_loss(pred_mask_vis_roi, self.gt_mask_vis_roi.to(dtype=pred_mask_vis_roi.dtype),
                                          p=1)
        else:
            raise NotImplementedError

    def relative_angle(self, pred_cam_R_m2c: torch.Tensor = None, degree: bool =False) -> torch.Tensor:
        """

        :param pred_cam_R_m2c: [..., 3, 3]
        :param degree: bool, use 360 degrees
        :return: [...]
        """
        pred_cam_R_m2c = _get_param(pred_cam_R_m2c, self.pred_cam_R_m2c)
        R_diff = pred_cam_R_m2c @ self.gt_cam_R_m2c.transpose(-2, -1)  # [..., 3, 3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # [...]
        angle = ((trace.clamp(-1., 3.) - 1.) * .5).acos()  # [...]
        if degree:
            angle *= 180. / torch.pi  # 360 degrees
        return angle  # [...]

    def relative_dist(self, pred_cam_t_m2c: torch.Tensor = None, cm: bool =False) -> torch.Tensor:
        """

        :param pred_cam_t_m2c: [.., 3]
        :param cm: bool, use cm instead of meter
        :return:
        """
        pred_cam_t_m2c = _get_param(pred_cam_t_m2c, self.pred_cam_t_m2c, lambda: self.get_pred_cam_t_m2c(store=True))
        dist = torch.linalg.vector_norm(pred_cam_t_m2c - self.gt_cam_t_m2c, ord=2, dim=-1)  # [...]
        if cm:
            dist *= 100.  # cm
        return dist  # [...]

    def visualize(self, return_figs: bool = False, pred_coord_3d_roi: torch.Tensor = None,
                  pred_mask_vis_roi: torch.Tensor = None, pred_cam_R_m2c: torch.Tensor = None,
                  pred_cam_t_m2c: torch.Tensor = None, max_samples: int = None) -> Optional[list[Figure]]:
        pred_coord_3d_roi_normalized = _get_param(
            (pred_coord_3d_roi, lambda x: self._get_coord_3d_roi_normalized(x)),
            self.pred_coord_3d_roi_normalized,
            (self.pred_coord_3d_roi, lambda x: self._get_coord_3d_roi_normalized(x))
        )
        pred_coord_3d_roi = _get_param(
            pred_coord_3d_roi, self.pred_coord_3d_roi,
            (self.pred_coord_3d_roi_normalized, lambda x: self._get_coord_3d_roi(x))
        )
        pred_mask_vis_roi = _get_param(pred_mask_vis_roi, self.pred_mask_vis_roi)
        pred_cam_R_m2c = _get_param(pred_cam_R_m2c, self.pred_cam_R_m2c)
        pred_cam_t_m2c = _get_param(pred_cam_t_m2c, self.pred_cam_t_m2c,
                                    (self.pred_cam_t_m2c_site, lambda: self.get_pred_cam_t_m2c(store=True)))
        figs = []
        for i in range(min(max_samples, len(self.obj_id)) if max_samples is not None else len(self.obj_id)):
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))

            utils.image_2d.draw_ax(axs[0, 0], self.img_roi[i])
            axs[0, 0].set_title('rendered image')

            if self.coord_2d_roi is not None:
                utils.image_2d.draw_ax(axs[1, 0], utils.image_2d.normalize_channel(self.coord_2d_roi[i]))
            else:
                axs[1, 0].set_aspect('equal')
            axs[1, 0].set_title('2D coord (relative)')

            if self.gt_coord_3d_roi is not None and self.obj_size is not None:
                utils.image_2d.draw_ax(axs[0, 1],
                    utils.transform_3d.normalize_coord_3d(self.gt_coord_3d_roi[i], self.obj_size[i]))
            else:
                axs[0, 1].set_aspect('equal')
            axs[0, 1].set_title('gt 3D coord (relative)')

            if pred_coord_3d_roi_normalized is not None:
                utils.image_2d.draw_ax(axs[1, 1], pred_coord_3d_roi_normalized[i].clamp(0., 1.))
            else:
                axs[1, 1].set_aspect('equal')
            axs[1, 1].set_title('pred 3D coord (relative)')

            if self.gt_mask_vis_roi is not None:
                if self.gt_mask_obj_roi is not None:
                    gt_mask_roi = torch.cat([self.gt_mask_obj_roi[i], self.gt_mask_vis_roi[i]], dim=-3)
                else:
                    gt_mask_roi = self.gt_mask_vis_roi[i].expand(2, -1, -1)
                utils.image_2d.draw_ax(axs[0, 2], gt_mask_roi)
            else:
                axs[0, 2].set_aspect('equal')
            axs[0, 2].set_title('gt mask (r-obj, g-vis)')

            if pred_mask_vis_roi is not None:
                utils.image_2d.draw_ax(axs[1, 2], pred_mask_vis_roi[i].clamp(0., 1.).expand(2, -1, -1))
            else:
                axs[1, 2].set_aspect('equal')
            axs[1, 2].set_title('pred mask vis')

            utils.transform_3d.show_pose(axs[0, 3], self.cam_K[i], self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i],
                                         self.obj_size[i], self.bbox[i], self.bbox_zoom_out_ratio, True)
            axs[0, 3].set_title('gt pose')
            if pred_cam_R_m2c is not None and pred_cam_t_m2c is not None:
                utils.transform_3d.show_pose(axs[1, 3], self.cam_K[i], pred_cam_R_m2c[i], pred_cam_t_m2c[i],
                                             self.obj_size[i], self.bbox[i], self.bbox_zoom_out_ratio, True)
            else:
                axs[1, 3].set_aspect('equal')
            axs[1, 3].set_title('pred pose')

            if pred_coord_3d_roi is not None and self.gt_coord_3d_roi is not None:
                diff = torch.linalg.vector_norm(pred_coord_3d_roi[i] * self.gt_mask_vis_roi[i] - self.gt_coord_3d_roi[i], dim=-3)
                utils.image_2d.draw_ax_diff(axs[2, 1], diff, thresh_min=0., thresh_max=1e-2, log_mode=False)
            else:
                axs[2, 1].set_aspect('equal')
            axs[2, 1].set_title('diff 3D coord (L2)')

            if pred_mask_vis_roi is not None and self.gt_mask_vis_roi is not None:
                diff = pred_mask_vis_roi[i] - self.gt_mask_vis_roi[i].to(dtype=pred_mask_vis_roi.dtype)
                utils.image_2d.draw_ax_diff(axs[2, 2], diff.abs(), thresh_min=1e-4, thresh_max=1., log_mode=True)
            else:
                axs[2, 2].set_aspect('equal')
            axs[2, 2].set_title('diff mask (abs)')

            if self.pnp_inlier_roi is not None:
                utils.image_2d.draw_ax(axs[2, 0], self.pnp_inlier_roi[i].expand(3, -1, -1))
            else:
                axs[2, 0].set_aspect('equal')
            axs[2, 0].set_title('pnp inlier')

            if self.pnp_cam_R_m2c is not None and self.pnp_cam_t_m2c is not None:
                utils.transform_3d.show_pose(axs[2, 3], self.cam_K[i], self.pnp_cam_R_m2c[i], self.pnp_cam_t_m2c[i],
                                             self.obj_size[i], self.bbox[i], self.bbox_zoom_out_ratio, True)
            else:
                axs[2, 3].set_aspect('equal')
            axs[2, 3].set_title('pnp pose')

            fig.tight_layout()
            if return_figs:
                figs.append(fig)
            else:
                plt.show()

        if self.dbg_img is not None:
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            utils.image_2d.draw_ax(ax, self.dbg_img[0], bboxes=self.bbox)
            for i in range(len(self.obj_id)):
                utils.transform_3d.show_pose(
                    ax, self.cam_K[i], self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i], self.obj_size[i])
            fig.tight_layout()
            if return_figs:
                figs.append(fig)
            else:
                plt.show()
        return figs if return_figs else None

    def clone(self, detach: bool = False):
        out = Sample()
        for key in [key for key in dir(self) if not key.startswith('__') and not callable(getattr(self, key))]:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                if detach:
                    setattr(out, key, value.detach())
                else:
                    setattr(out, key, value.clone())
            else:
                setattr(out, key, copy.deepcopy(value))
        return out
