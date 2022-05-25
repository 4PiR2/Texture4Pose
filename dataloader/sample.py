import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataloader.obj_mesh import ObjMesh
from utils.const import debug_mode, bbox_zoom_out
from utils.image_2d import normalize_channel, draw_ax, lp_loss
from utils.transform_3d import normalize_coord_3d, ransac_pnp, show_pose


class Sample:
    def __init__(self, obj_id=None, obj_size=None, cam_K=None, gt_cam_R_m2c=None, gt_cam_t_m2c=None,
                 gt_cam_t_m2c_site=None, coord_2d_roi=None, gt_coord_3d_roi=None, gt_mask_vis_roi=None,
                 gt_mask_obj_roi=None, img_roi=None, dbg_img=None, bbox=None, gt_bbox_vis=None, gt_bbox_obj=None):
        self.obj_id: torch.Tensor = obj_id
        self.obj_size: torch.Tensor = obj_size
        self.cam_K: torch.Tensor = cam_K
        self.gt_cam_R_m2c: torch.Tensor = gt_cam_R_m2c
        self.gt_cam_t_m2c: torch.Tensor = gt_cam_t_m2c
        self.gt_cam_t_m2c_site: torch.Tensor = gt_cam_t_m2c_site
        self.coord_2d_roi: torch.Tensor = coord_2d_roi
        self.gt_coord_3d_roi: torch.Tensor = gt_coord_3d_roi
        self.gt_mask_vis_roi: torch.Tensor = gt_mask_vis_roi
        self.gt_mask_obj_roi: torch.Tensor = gt_mask_obj_roi
        self.img_roi: torch.Tensor = img_roi
        self.dbg_img: torch.Tensor = dbg_img
        self.bbox: torch.Tensor = bbox
        self.gt_bbox_vis: torch.Tensor = gt_bbox_vis
        self.gt_bbox_obj: torch.Tensor = gt_bbox_obj

    @classmethod
    def collate(cls, batch: list):
        keys = [key for key in dir(batch[0]) if not key.startswith('__') and not callable(getattr(batch[0], key))]
        out = cls()
        for key in keys:
            if key in ['dataset']:
                setattr(out, key, getattr(batch[0], key))
            else:
                if key.startswith('dbg_') and debug_mode == False:
                    continue
                setattr(out, key, torch.cat([getattr(b, key) for b in batch], dim=0))
        return out

    def sanity_check(self) -> tuple[torch.Tensor, torch.Tensor]:
        pred_cam_R_m2c, pred_cam_t_m2c = ransac_pnp(self.gt_coord_3d_roi, self.coord_2d_roi, self.gt_mask_vis_roi)
        return pred_cam_R_m2c, pred_cam_t_m2c

    def pm_loss(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :return: [N]
        """
        n = len(self.obj_id)
        loss = torch.empty(n, device=pred_cam_R_m2c.device)
        for i in range(n):
            obj = objects_eval[int(self.obj_id[i])]
            loss[i] = obj.average_distance(pred_cam_R_m2c[i], self.gt_cam_R_m2c[i], p=1)
        return loss

    def add_score(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor,
                  div_diameter: bool = True) -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :param div_diameter: bool
        :return: [N]
        """
        n = len(self.obj_id)
        add = torch.empty(n, device=pred_cam_R_m2c.device)
        for i in range(n):
            obj = objects_eval[int(self.obj_id[i])]
            add[i] = obj.average_distance(pred_cam_R_m2c[i], self.gt_cam_R_m2c[i], pred_cam_t_m2c[i],
                                          self.gt_cam_t_m2c[i])
            if div_diameter:
                add[i] /= obj.diameter
        return add

    def proj_dist(self, objects_eval: dict[int, ObjMesh], pred_cam_R_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor) \
            -> torch.Tensor:
        """
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :return: [N]
        """
        n = len(self.obj_id)
        dist = torch.empty(n, device=pred_cam_R_m2c.device)
        for i in range(n):
            obj = objects_eval[int(self.obj_id[i])]
            dist[i] = obj.average_projected_distance(self.cam_K[i], pred_cam_R_m2c[i], self.gt_cam_R_m2c[i],
                                                     pred_cam_t_m2c[i], self.gt_cam_t_m2c[i])
        return dist

    def t_site_center_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.linalg.vector_norm(self.gt_cam_t_m2c_site[:, :2] - pred_cam_t_m2c_site[:, :2], ord=1, dim=-1)

    def t_site_depth_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.abs(self.gt_cam_t_m2c_site[:, 2] - pred_cam_t_m2c_site[:, 2])

    def coord_3d_loss(self, pred_coord_3d_roi_normalized: torch.Tensor) -> torch.Tensor:
        """
        :param pred_coord_3d_roi_normalized: [..., 3, H, W]
        :return: [...]
        """
        gt_coord_3d_roi_normalized = normalize_coord_3d(self.gt_coord_3d_roi, self.obj_size)
        return lp_loss((pred_coord_3d_roi_normalized - gt_coord_3d_roi_normalized) * self.gt_mask_vis_roi, p=1)

    def mask_loss(self, pred_mask_vis_roi: torch.Tensor) -> torch.Tensor:
        """
        :param pred_mask_vis_roi: [..., 1, H, W]
        :return: [...]
        """
        return lp_loss(pred_mask_vis_roi, self.gt_mask_vis_roi.to(dtype=pred_mask_vis_roi.dtype), p=1)

    def relative_angle(self, pred_cam_R_m2c: torch.Tensor, degree=False) -> torch.Tensor:
        """

        :param pred_cam_R_m2c: [..., 3, 3]
        :param degree: bool, use 360 degrees
        :return: [...]
        """
        R_diff = pred_cam_R_m2c @ self.gt_cam_R_m2c.transpose(-2, -1)  # [..., 3, 3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # [...]
        angle = ((trace.clamp(-1., 3.) - 1.) * .5).acos()  # [...]
        if degree:
            angle *= 180. / torch.pi  # 360 degrees
        return angle  # [...]

    def relative_dist(self, pred_cam_t_m2c: torch.Tensor, cm=False) -> torch.Tensor:
        """

        :param pred_cam_t_m2c: [.., 3]
        :param cm: bool, use cm instead of meter
        :return:
        """
        dist = torch.linalg.vector_norm(pred_cam_t_m2c - self.gt_cam_t_m2c, ord=2, dim=-1)  # [...]
        if cm:
            dist *= 100.  # cm
        return dist  # [...]

    def visualize(self, pred_coord_3d_roi: torch.Tensor = None, pred_mask_roi: torch.Tensor = None,
                  pred_cam_R_m2c: torch.Tensor = None, pred_cam_t_m2c: torch.Tensor = None) -> None:
        for i in range(len(self.obj_id)):
            fig, axs = plt.subplots(4, 2, figsize=(6, 14))
            draw_ax(axs[0, 0], self.img_roi[i])
            axs[0, 0].set_title('rendered image')
            draw_ax(axs[0, 1], normalize_channel(self.coord_2d_roi[i]))
            axs[0, 1].set_title('2D coord (norm)')
            draw_ax(axs[1, 0], normalize_coord_3d(self.gt_coord_3d_roi[i], self.obj_size[i]))
            axs[1, 0].set_title('gt 3D coord (norm)')
            if pred_coord_3d_roi is not None:
                draw_ax(axs[1, 1], normalize_coord_3d(pred_coord_3d_roi[i], self.obj_size[i]).clamp(0., 1.))
            else:
                axs[1, 1].set_aspect('equal')
            axs[1, 1].set_title('pred 3D coord (norm)')
            draw_ax(axs[2, 0], torch.cat([self.gt_mask_obj_roi[i], self.gt_mask_vis_roi[i]], dim=0))
            axs[2, 0].set_title('gt mask (r-obj, g-vis)')
            if pred_mask_roi is not None:
                draw_ax(axs[2, 1], pred_mask_roi[i])
            else:
                axs[2, 1].set_aspect('equal')
            axs[2, 1].set_title('pred mask')
            show_pose(axs[3, 0], self.cam_K[i], self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i], self.obj_size[i],
                      self.bbox[i], bbox_zoom_out, True)
            axs[3, 0].set_title('gt pose')
            if pred_cam_R_m2c is not None and pred_cam_t_m2c is not None:
                show_pose(axs[3, 1], self.cam_K[i], pred_cam_R_m2c[i], pred_cam_t_m2c[i], self.obj_size[i],
                          self.bbox[i], bbox_zoom_out, True)
            else:
                axs[3, 1].set_aspect('equal')
            axs[3, 1].set_title('pred pose')
            plt.show()

        if debug_mode:
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            draw_ax(ax, self.dbg_img[0], bboxes=self.bbox)
            for i in range(len(self.obj_id)):
                show_pose(ax, self.cam_K[i], self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i], self.obj_size[i])
            plt.show()
