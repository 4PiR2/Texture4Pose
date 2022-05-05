import torch
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init

from dataloader.Sample import Sample
from utils.const import pnp_input_size, gdr_mode
from utils.transform import calculate_bbox_crop, t_site_to_t


class ConvPnPNet(nn.Module):
    def __init__(self, nIn, featdim=128, rot_dim=6, num_layers=3, num_gn_groups=32):
        """
        Args:
            nIn: input feature channel
        """
        super().__init__()

        assert num_layers >= 3, num_layers
        # nIn += 64
        conv_layers = []
        for i in range(3):
            in_channels = nIn if i == 0 else featdim
            stride = 2 if i < 3 else 1
            conv_layers.append(nn.Conv2d(in_channels, featdim, kernel_size=3, stride=stride, padding=1, bias=False))
            conv_layers.append(nn.GroupNorm(num_gn_groups, featdim))
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        fc_layers = [nn.Flatten(),
                     nn.Linear(featdim * 8 * 8, 1024),
                     nn.LeakyReLU(0.1, inplace=True),
                     nn.Linear(1024, 256),
                     nn.LeakyReLU(0.1, inplace=True)]
        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc_R = nn.Linear(256, rot_dim)
        self.fc_t = nn.Linear(256, 3)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_R, std=0.01)
        normal_init(self.fc_t, std=0.01)
    
    def load_pretrain(self, gdr_pth_path):
        params = torch.load(gdr_pth_path)['model']
        pnp_params = {}
        for k in params:
            if k.startswith('pnp_net.'):
                pnp_params[k] = nn.Parameter(params[k])
        with torch.no_grad():
            self.conv_layers[0].weight = nn.Parameter(pnp_params['pnp_net.features.0.weight'][:, :5])
            self.conv_layers[1].weight = pnp_params['pnp_net.features.1.weight']
            self.conv_layers[1].bias = pnp_params['pnp_net.features.1.bias']
            self.conv_layers[3].weight = pnp_params['pnp_net.features.3.weight']
            self.conv_layers[4].weight = pnp_params['pnp_net.features.4.weight']
            self.conv_layers[4].bias = pnp_params['pnp_net.features.4.bias']
            self.conv_layers[6].weight = pnp_params['pnp_net.features.6.weight']
            self.conv_layers[7].weight = pnp_params['pnp_net.features.7.weight']
            self.conv_layers[7].bias = pnp_params['pnp_net.features.7.bias']
            self.fc_layers[1].weight = pnp_params['pnp_net.fc1.weight']
            self.fc_layers[1].bias = pnp_params['pnp_net.fc1.bias']
            self.fc_layers[3].weight = pnp_params['pnp_net.fc2.weight']
            self.fc_layers[3].bias = pnp_params['pnp_net.fc2.bias']
            self.fc_R.weight = pnp_params['pnp_net.fc_r.weight']
            self.fc_R.bias = pnp_params['pnp_net.fc_r.bias']
            self.fc_t.weight = pnp_params['pnp_net.fc_t.weight']
            self.fc_t.bias = pnp_params['pnp_net.fc_t.bias']

    def forward(self, sample: Sample):
        c2d = sample.coor2d
        if gdr_mode:
            c2d = c2d.permute(0, 2, 3, 1)  # [N, H, W, 2(XY)]
            c2d = torch.cat([c2d, torch.ones_like(c2d[..., :1])], dim=-1)[..., None]  # [N, H, W, 3(XY1), 1]
            c2d = sample.cam_K[None, None, None] @ c2d
            c2d = c2d[..., :2, 0].permute(0, 3, 1, 2)  # [N, 2(XY), H, W]
            c2d /= torch.tensor([640, 480]).to(c2d.device)[None, ..., None, None]

        x = torch.cat([sample.gt_coor3d, c2d], dim=1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        pred_cam_R_m2c_6d, pred_cam_t_m2c_site = self.fc_R(x), self.fc_t(x)
        pred_cam_R_m2c = rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        if gdr_mode:
            pred_cam_R_m2c = pred_cam_R_m2c.transpose(-2, -1)
        crop_size, *_ = calculate_bbox_crop(sample.bbox)
        pred_cam_t_m2c = t_site_to_t(pred_cam_t_m2c_site, sample.bbox,
                                     pnp_input_size / crop_size, sample.cam_K)
        return pred_cam_R_m2c, pred_cam_t_m2c, pred_cam_t_m2c_site
