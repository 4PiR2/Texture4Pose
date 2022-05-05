import torch
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from dataloader.Sample import Sample
from utils.const import pnp_input_size, gdr_mode
from utils.transform import calculate_bbox_crop, t_site_to_t
from utils.weight_init import normal_init, constant_init


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

        if gdr_mode:
            self.gdr_layer = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
            weight = torch.eye(5)
            bias = torch.zeros(5)
            K = torch.Tensor([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899]])
            K[0] /= 640.
            K[1] /= 480.
            weight[3:, 3:] = K[:, :2]
            bias[3:] = K[:, -1]
            self.gdr_layer.weight = nn.Parameter(weight[..., None, None])
            self.gdr_layer.bias = nn.Parameter(bias)
        else:
            self.gdr_layer = None
    
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

    def forward(self, sample: Sample, pred_coor3d=None):
        if pred_coor3d is None:
            pred_coor3d = sample.gt_coor3d
        elif gdr_mode:
            pred_coor3d = (pred_coor3d - .5) * sample.obj_size[..., None, None]

        x = torch.cat([pred_coor3d, sample.coor2d], dim=1)

        if self.gdr_layer is not None:
            x = self.gdr_layer(x)  # coor2d \in [0, 1]

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
