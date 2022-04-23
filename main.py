import cv2
import torch
from matplotlib import pyplot as plt, patches
from pytorch3d.transforms import so3_relative_angle

from dataloader.BOPDataset import BOPDataset
from utils.const import lmo_objects, device, plot_colors


dataset = BOPDataset(obj_list=lmo_objects, path='data/BOP/lmo', device=device)

scene_id_test = 3

result = dataset[scene_id_test]


def debug_show(img_1, bg_255=None, mask=None, bboxes=None):
    img_255 = img_1.permute(1, 2, 0) * 255
    img_255 = img_255[..., :3]
    if bg_255 is not None:
        if mask is not None:
            if mask.shape[-1] != 1:
                mask = mask.unsqueeze(-1)
            mask = mask.bool()
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
            bboxes = bboxes.unsqueeze(0)
        bboxes = bboxes.cpu().numpy()
        for i in range(len(bboxes)):
            add_bbox(ax, *bboxes[i], text=str(i), color=plot_colors[i % len(plot_colors)])
    plt.show()


# for i in range(len(result['gt_obj_id'])):
#     debug_show(result['imgs'][0][i])
#     debug_show(result['gt_coor3d'][i])
#     debug_show(result['gt_mask_vis'][i])
#     debug_show(result['gt_mask_obj'][i])
#     debug_show(torch.cat([result['gt_coor2d'][i], torch.zeros((1, 64, 64), device=device)], dim=0))
# debug_show(result['dbg_img'][0], bboxes=result['dbg_bbox'])

i = 1
coor3d = result['gt_coor3d'][i].permute(1, 2, 0)
coor2d = result['gt_coor2d'][i].permute(1, 2, 0)
mask = result['gt_mask_vis'][i].squeeze()

gt_K = torch.Tensor([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]).to(device)

x = coor3d[mask]
y = coor2d[mask]

_, R_exp, T, _ = cv2.solvePnPRansac(x.cpu().numpy(), y.cpu().numpy(), gt_K.cpu().numpy(), None)
R, _ = cv2.Rodrigues(R_exp)
R, T = torch.Tensor(R).to(device), torch.Tensor(T).to(device).flatten()

gt_R = result['gt_cam_R_m2c'][i]
gt_T = result['gt_cam_t_m2c'][i]

angle = so3_relative_angle(R[None], gt_R[None])
dist = torch.norm(T - gt_T)

gt_proj = (x @ gt_R.T + gt_T) @ gt_K.T
gt_proj = gt_proj[:, :2] / gt_proj[:, 2:]

proj = (x @ R.T + T) @ gt_K.T
proj = proj[:, :2] / proj[:, 2:]

a = 0
