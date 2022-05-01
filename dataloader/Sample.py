import torch
from matplotlib import pyplot as plt, patches

from utils.const import debug_mode, plot_colors


class Sample:
    def __init__(self, obj_id=None, cam_K=None, gt_cam_R_m2c=None, gt_cam_t_m2c=None, coor2d=None, gt_coor3d=None,
                 gt_mask_vis=None, gt_mask_obj=None, img=None, dbg_img=None, dbg_bbox=None):
        self.obj_id = obj_id
        self.cam_K = cam_K
        self.gt_cam_R_m2c = gt_cam_R_m2c
        self.gt_cam_t_m2c = gt_cam_t_m2c
        self.coor2d = coor2d
        self.gt_coor3d = gt_coor3d
        self.gt_mask_vis = gt_mask_vis
        self.gt_mask_obj = gt_mask_obj
        self.img = img
        self.dbg_img = dbg_img
        self.dbg_bbox = dbg_bbox

    # @staticmethod
    @classmethod
    def collate(cls, batch):
        keys = [key for key in dir(batch[0]) if not key.startswith('__') and not callable(getattr(batch[0], key))]
        out = cls()
        for key in keys:
            if key in ['cam_K']:
                setattr(out, key, getattr(batch[0], key))
            else:
                if key.startswith('dbg_') and debug_mode == False:
                    continue
                setattr(out, key, torch.cat([getattr(b, key) for b in batch], dim=0))
        return out

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

        for i in range(len(self.dbg_img)):
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            draw(ax, self.dbg_img[i], bboxes=self.dbg_bbox)
            plt.show()
