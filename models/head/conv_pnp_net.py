import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import utils.image_2d
import utils.transform_3d
import utils.weight_init


class ConvPnPNet(nn.Module):
    def __init__(self, in_channels=3+2, featdim=128, num_layers=3, num_gn_groups=32):
        super().__init__()

        assert num_layers >= 3, num_layers
        rot_dim = 6
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

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        pred_cam_R_m2c_6d, pred_cam_t_m2c_site = self.fc_R(x), self.fc_t(x)
        return pred_cam_R_m2c_6d, pred_cam_t_m2c_site
