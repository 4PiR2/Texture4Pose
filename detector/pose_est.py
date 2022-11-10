import pickle

import torch
from tqdm import tqdm

import config.const as cc
import utils.io
import utils.image_2d
import utils.transform_3d


def get_t4p_model():
    from utils.config import Config
    from dataloader.data_module import LitDataModule
    from models.main_model import MainModel

    ckpt_path = '/data/lightning_logs_archive/104/104_siren_long/version_341/checkpoints/epoch=0142-val_metric=0.3394.ckpt'

    def setup_t4p(args=None) -> Config:
        cfg = Config.fromfile('config/top.py')
        if args is not None:
            cfg.merge_from_dict(args)
        return cfg

    cfg_t4p = setup_t4p()
    datamodule = LitDataModule(cfg_t4p)

    if ckpt_path is not None:
        t4p_model = MainModel.load_from_checkpoint(
            ckpt_path, strict=True, cfg=cfg_t4p,
            objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval
        )
    else:
        t4p_model = MainModel(cfg_t4p, datamodule.dataset.objects, datamodule.dataset.objects_eval)

    t4p_model = t4p_model.to(cc.device)
    t4p_model.eval()
    return t4p_model


def est_pose():
    # fig = plt.figure(figsize=(4032 * 1e-2, 3024 * 1e-2))
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # fig.add_axes(ax)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.set_axis_off()
    # fig.patch.set_alpha(0.)
    # ax.patch.set_alpha(0.)
    # ax.imshow(im)
    # utils.transform_3d.show_pose_mesh_105(ax, cc.i12P_cam_K.to(x3d.device)[0], pred_cam_R_m2c[0], pred_cam_t_m2c[0])
    # plt.savefig('/home/user/Desktop/1.svg')
    # plt.show()
    pass


if __name__ == '__main__':
    t4p_model = get_t4p_model()
    t4p_model.eval()
    with open('outputs/detections.pkl', 'rb') as f:
        detections = pickle.load(f)
    img_path_list = utils.io.list_img_from_dir('/data/real_exp/i12P_26mm/000104/siren', ext='heic')
    outputs_list = []
    for img_path, detection in tqdm(zip(img_path_list, detections)):
        fields = detection['instances']._fields
        scores = fields['scores']
        pred_classes = fields['pred_classes']
        if len(scores) < 1:
            outputs_list.append(None)
            continue
        x1, y1, x2, y2 = fields['pred_boxes'].tensor[0]
        x = (x1 + x2) * .5
        y = (y1 + y2) * .5
        s = max(x2 - x1, y2 - y1)

        im = utils.io.imread(img_path, opencv_bgr=False)
        img = torch.tensor(im).permute(2, 0, 1).to(s.device) / 255.
        cim = utils.image_2d.crop_roi(img, torch.stack([x, y, s, s], dim=-1)[None], s * 1.5, 256, 'bilinear')
        with torch.no_grad():
            features = t4p_model.resnet_backbone(cim)
            features = t4p_model.up_sampling_backbone(features)
            x3d = t4p_model.coord_3d_head(features)
            x3d = (x3d - .5) * .1
            w2d_raw, log_weight_scale = t4p_model.secondary_head(features)
            N, _, H, W = w2d_raw.shape
            x3d = x3d.permute(0, 2, 3, 1).reshape(N, -1, 3)
            w2d = w2d_raw.permute(0, 2, 3, 1).reshape(N, -1, 2)
            x2d = torch.stack(torch.meshgrid(torch.linspace(x - s * .5 * 1.5, x + s * .5 * 1.5, 64),
                                             torch.linspace(y - s * .5 * 1.5, y + s * .5 * 1.5, 64), indexing='xy'))
            x2d = x2d.to(x3d.device)[None].permute(0, 2, 3, 1).reshape(N, -1, 2)
            cam_K = cc.i12P_cam_K.to(x3d.device)

            pose_opt = t4p_model.pnp_net.forward_test(x3d, x2d, w2d, log_weight_scale, cam_K, fast_mode=False)
            outputs_list.append(pose_opt)

    with open('outputs/poses.pkl', 'wb') as f:
        pickle.dump(outputs_list, f)
    a = 0
