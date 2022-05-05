import torch
import torch.nn as nn

from dataloader.Sample import Sample


class GDRN(nn.Module):
    def __init__(self, backbone, rot_head_net, pnp_net):
        super().__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net

    def forward(self, sample: Sample):
        # x.shape [bs, 3, 256, 256]
        features = self.backbone(sample.img)
        mask, coor_3d, region = self.rot_head_net(features)

        pred_cam_R_m2c, pred_cam_t_m2c, pred_cam_t_m2c_site = self.pnp_net(sample, coor_3d)
        return pred_cam_R_m2c, pred_cam_t_m2c, pred_cam_t_m2c_site

    def load_pretrain(self, gdr_pth_path):
        state_dict = torch.load(gdr_pth_path)['model']
        self.load_state_dict(state_dict, strict=False)
        self.pnp_net.load_pretrain(gdr_pth_path)
