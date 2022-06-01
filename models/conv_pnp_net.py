import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import utils.image_2d
import utils.transform_3d
import utils.weight_init


class ConvPnPNet(nn.Module):
    def __init__(self, in_channels, featdim=128, rot_dim=6, num_layers=3, num_gn_groups=32):
        super().__init__()

        assert num_layers >= 3, num_layers
        conv_layers = []
        for i in range(3):
            num_in = in_channels if i == 0 else featdim
            stride = 2 if i < 3 else 1
            conv_layers.append(nn.Conv2d(num_in, featdim, kernel_size=3, stride=stride, padding=1, bias=False))
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
                utils.weight_init.normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                utils.weight_init.constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                utils.weight_init.normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                utils.weight_init.normal_init(m, std=0.001)
        utils.weight_init.normal_init(self.fc_R, std=0.01)
        utils.weight_init.normal_init(self.fc_t, std=0.01)

    def load_pretrain(self, gdr_pth_path):
        params = torch.load(gdr_pth_path)['model']
        pnp_params = {}
        for k in params:
            if k.startswith('pnp_net.'):
                pnp_params[k] = nn.Parameter(params[k])
        with torch.no_grad():
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

    def forward(self, pred_coord_3d_roi: torch.Tensor, coord_2d_roi: torch.Tensor, mask: torch.Tensor):
        x = torch.cat([pred_coord_3d_roi, coord_2d_roi, mask], dim=1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        pred_cam_R_m2c_6d, pred_cam_t_m2c_site = self.fc_R(x), self.fc_t(x)
        return pred_cam_R_m2c_6d, pred_cam_t_m2c_site
