import os.path
import pickle

from matplotlib import pyplot as plt
import pytorch3d.transforms
import torch
from tqdm import tqdm

import config.const as cc
import utils.io
import utils.transform_3d


if __name__ == '__main__':
    oid = 105
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'holding')
    with open(os.path.join(root_dir, 'poses.pkl'), 'rb') as f:
        poses = pickle.load(f)
    img_path_list = utils.io.list_img_from_dir(os.path.join(root_dir, 'orig'), ext='png')
    outputs_list = []
    for img_path, pose in tqdm(zip(img_path_list, poses)):
        im = utils.io.imread(img_path, opencv_bgr=False)

        H, W = im.shape[:2]
        fig = plt.figure(figsize=(W * 1e-2, H * 1e-2))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_axis_off()
        fig.patch.set_alpha(0.)
        ax.patch.set_alpha(0.)
        ax.imshow(im)
        if pose is not None:
            pred_cam_t_m2c = pose[:, :3]
            pred_cam_R_m2c = pytorch3d.transforms.quaternion_to_matrix(pose[:, 3:])
            utils.transform_3d.show_pose_mesh(
                ax, cc.video_cam_K.to(pose.device)[0], pred_cam_R_m2c[0], pred_cam_t_m2c[0],
                obj_id=oid, no_diag_edges=oid == 104)
        ax.set_xlim(-.5, W - .5)
        ax.set_ylim(H - .5, -.5)
        plt.savefig(os.path.join(root_dir, 'pose', f'{img_path.split("/")[-1].split(".")[0]}.png'))
        # plt.show()
        plt.close()
