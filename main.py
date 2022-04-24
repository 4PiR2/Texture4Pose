import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from pytorch3d.transforms import so3_relative_angle
from torchvision.transforms import transforms as T

from dataloader.BOPDataset import BOPDataset
from utils.const import lmo_objects, device, plot_colors

if __name__ == '__main__':
    composed = T.Compose([T.RandomGrayscale(p=0.1)])

    dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', transform=composed, device=device)
    scene_id_test = 3
    result = dataset[scene_id_test]

    def debug_show(img_1, bg_1=None, mask=None, bboxes=None):
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
        plt.show()

    for i in range(len(result['obj_id'])):
        debug_show(result['imgs'][0][i])
        debug_show(result['gt_coor3d'][i])
        debug_show(result['gt_mask_vis'][i])
        debug_show(result['gt_mask_obj'][i])
        debug_show(result['coor2d'][i])
    debug_show(result['dbg_imgs'][0], bboxes=result['dbg_bbox'])

    for i in range(1, 2):
        mask = result['gt_mask_vis'][i].squeeze()
        x = result['gt_coor3d'][i].permute(1, 2, 0)[mask]
        y = result['coor2d'][i].permute(1, 2, 0)[mask]

        gt_K = result['gt_cam_K']
        gt_R = result['gt_cam_R_m2c'][i]
        gt_t = result['gt_cam_t_m2c'][i]

        _, pred_R_exp, pred_t, _ = cv2.solvePnPRansac(x.cpu().numpy(), y.cpu().numpy(), np.eye(3), None)
        pred_R, _ = cv2.Rodrigues(pred_R_exp)
        pred_R, pred_t = torch.Tensor(pred_R).to(device), torch.Tensor(pred_t).to(device).flatten()

        angle = so3_relative_angle(pred_R[None], gt_R[None])
        dist = torch.norm(pred_t - gt_t)

        gt_proj = (x @ gt_R.T + gt_t) @ gt_K.T
        gt_proj = gt_proj[:, :2] / gt_proj[:, 2:]

        pred_proj = (x @ pred_R.T + pred_t) @ gt_K.T
        pred_proj = pred_proj[:, :2] / pred_proj[:, 2:]

        a = 0


# def poses_from_random(dataset, num_obj):
#     device = dataset.device
#     euler_angles = (2. * torch.pi) * torch.rand((num_obj, 3), device=device)
#     Rs = euler_angles_to_matrix(euler_angles, 'ZYX')
#
#     objects = dataset.objects
#     selected_obj = [obj_id for obj_id in objects]
#     random.shuffle(selected_obj)
#     selected_obj = selected_obj[:num_obj]
#     selected_obj.sort()
#     radii = torch.tensor([objects[obj_id].radius for obj_id in selected_obj], device=device)
#     centers = torch.stack([objects[obj_id].center for obj_id in selected_obj], dim=0)
#     triu_indices = torch.triu_indices(num_obj, num_obj, 1)
#     mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]
#
#     flag = False
#     while not flag:
#         positions = torch.rand((num_obj, 3), device=device)\
#                     * torch.tensor((.5, .5, .5), device=device) + torch.tensor((-.25, -.25, 1.), device=device)
#         flag = (F.pdist(positions) >= mdist).all()
#     positions -= centers
#
#     poses = []
#     for i in range(num_obj):
#         poses.append({'obj_id': selected_obj[i], 'cam_R_m2c': Rs[i], 'cam_t_m2c': positions[i]})
#     return poses
