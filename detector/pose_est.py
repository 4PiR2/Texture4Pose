import os.path
import pickle

import torch
from tqdm import tqdm

import config.const as cc
import utils.get_model
import utils.io
import utils.image_2d
import utils.transform_3d


if __name__ == '__main__':
    oid = 105

    ckpt_path = os.path.join('weights', f'{oid}sa.ckpt')
    t4p_model, _ = utils.get_model.get_model(ckpt_path=ckpt_path)
    t4p_model.eval()

    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'rolling')

    with open(os.path.join(root_dir, 'detections.pkl'), 'rb') as f:
        detections = pickle.load(f)
    img_path_list = utils.io.list_img_from_dir(os.path.join(root_dir, 'orig'), ext='png')
    outputs_list = []
    for img_path, detection in tqdm(zip(img_path_list, detections)):
        fields = detection['instances']._fields
        scores = fields['scores']
        pred_classes = fields['pred_classes']
        if len(scores) < 1 or scores[0] < .01:
            outputs_list.append(None)
            continue
        x1, y1, x2, y2 = fields['pred_boxes'].tensor[0]
        # x1, y1, x2, y2 = torch.tensor([2844., 1144., 3500., 1700.], device=x1.device)
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
            cam_K = cc.video_cam_K.to(x3d.device)

            pose_opt = t4p_model.pnp_net.forward_test(x3d, x2d, w2d, log_weight_scale, cam_K, fast_mode=False)
            outputs_list.append(pose_opt)

    with open(os.path.join(root_dir, 'poses.pkl'), 'wb') as f:
        pickle.dump(outputs_list, f)
    a = 0
