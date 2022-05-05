import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from pytorch3d.ops import efficient_pnp
from pytorch3d.transforms import so3_relative_angle

from dataloader.ObjMesh import ObjMesh
from utils.const import debug_mode, plot_colors


class Sample:
    def __init__(self, obj_id=None, cam_K=None, gt_cam_R_m2c=None, gt_cam_t_m2c=None,
                 coor2d=None, gt_coor3d=None, gt_mask_vis=None, gt_mask_obj=None, img=None,
                 dbg_img=None, bbox=None, gt_cam_t_m2c_site=None, obj_size=None):
        self.obj_id: torch.Tensor = obj_id
        self.obj_size: torch.Tensor = obj_size
        self.cam_K: torch.Tensor = cam_K
        self.gt_cam_R_m2c: torch.Tensor = gt_cam_R_m2c
        self.gt_cam_t_m2c: torch.Tensor = gt_cam_t_m2c
        self.gt_cam_t_m2c_site: torch.Tensor = gt_cam_t_m2c_site
        self.coor2d: torch.Tensor = coor2d
        self.gt_coor3d: torch.Tensor = gt_coor3d
        self.gt_mask_vis: torch.Tensor = gt_mask_vis
        self.gt_mask_obj: torch.Tensor = gt_mask_obj
        self.img: torch.Tensor = img
        self.dbg_img: torch.Tensor = dbg_img
        self.bbox: torch.Tensor = bbox

    # @staticmethod
    @classmethod
    def collate(cls, batch):
        keys = [key for key in dir(batch[0]) if not key.startswith('__') and not callable(getattr(batch[0], key))]
        out = cls()
        for key in keys:
            if key in ['dataset', 'cam_K']:
                setattr(out, key, getattr(batch[0], key))
            else:
                if key.startswith('dbg_') and debug_mode == False:
                    continue
                setattr(out, key, torch.cat([getattr(b, key) for b in batch], dim=0))
        return out

    def sanity_check(self):
        pred_cam_R_m2c = torch.empty_like(self.gt_cam_R_m2c)
        pred_cam_t_m2c = torch.empty_like(self.gt_cam_t_m2c)
        for i in range(len(self.obj_id)):
            mask = self.gt_mask_vis[i].squeeze()
            x = self.gt_coor3d[i].permute(1, 2, 0)[mask]
            y = self.coor2d[i].permute(1, 2, 0)[mask]
            # sol = efficient_pnp(x[None], y[None])
            # pred_R2, pred_t2 = sol.R[0].T, sol.T[0]
            _, pred_R_exp, pred_t, _ = cv2.solvePnPRansac(x.cpu().numpy(), y.cpu().numpy(), np.eye(3), None,
                                                          reprojectionError=.01)
            pred_R, _ = cv2.Rodrigues(pred_R_exp)
            device = self.obj_id.device
            pred_cam_R_m2c[i] = torch.Tensor(pred_R).to(device)
            pred_cam_t_m2c[i] = torch.Tensor(pred_t).to(device).flatten()
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
            obj_id = int(self.obj_id[i])
            loss[i] = objects_eval[obj_id].point_match_error(pred_cam_R_m2c[i], self.gt_cam_R_m2c[i])
        return loss

    def t_site_center_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.norm(self.gt_cam_t_m2c_site[:, :2] - pred_cam_t_m2c_site[:, :2], p=1, dim=-1)

    def t_site_depth_loss(self, pred_cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        """
        :param pred_cam_t_m2c_site: [N, 3]
        :return: [N]
        """
        return torch.abs(self.gt_cam_t_m2c_site[:, 2] - pred_cam_t_m2c_site[:, 2])

    def relative_angle(self, pred_cam_R_m2c: torch.Tensor) -> torch.Tensor:
        return so3_relative_angle(pred_cam_R_m2c, self.gt_cam_R_m2c, eps=1.)

    def relative_dist(self, pred_cam_t_m2c: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred_cam_t_m2c - self.gt_cam_t_m2c, p=2, dim=-1)


    def visualize(self):
        def draw(ax, img_1, bg_1=None, mask=None, bboxes=None):
            img_255 = img_1.permute(1, 2, 0)[..., :3] * 255
            if img_255.shape[-1] == 2:
                img_255 = torch.cat([img_255, torch.zeros_like(img_255[..., :1])], dim=-1)
            if bg_1 is not None:
                bg_255 = bg_1.permute(1, 2, 0)[..., :3] * 255
                if mask is not None:
                    mask = mask.squeeze()[..., None].bool()
                    img_255 = img_255 * mask + bg_255 * ~mask
                else:
                    img_255 = img_255 * 0.5 + bg_255 * 0.5

            if ax is None:
                fig = plt.figure()
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                fig.add_axes(ax)
            ax.imshow(img_255.cpu().numpy().astype('uint8'))

            if bboxes is not None:
                def add_bbox(ax, x, y, w, h, text=None, color='red'):
                    rect = patches.Rectangle((x - w * .5, y - h * .5), w, h,
                                             linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y, text, color=color, size=12, ha='center', va='center')

                if bboxes.dim() < 2:
                    bboxes = bboxes[None]
                bboxes = bboxes.cpu().numpy()
                for i in range(len(bboxes)):
                    add_bbox(ax, *bboxes[i], text=str(i), color=plot_colors[i % len(plot_colors)])
            return ax

        for i in range(len(self.obj_id)):
            fig, axs = plt.subplots(2, 2)
            draw(axs[0, 0], self.img[i])
            draw(axs[0, 1], self.gt_coor3d[i])
            draw(axs[1, 0], torch.cat([self.gt_mask_obj[i], self.gt_mask_vis[i]], dim=0))
            draw(axs[1, 1], self.coor2d[i])
            plt.show()

        if debug_mode:
            for i in range(len(self.dbg_img)):
                fig = plt.figure()
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                fig.add_axes(ax)
                draw(ax, self.dbg_img[i], bboxes=self.bbox)
                plt.show()
