import torch
from pytorch3d.transforms import rotation_6d_to_matrix
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from dataloader.Sample import Sample
from utils.const import pnp_input_size, gdr_mode
from utils.transform import calc_bbox2d_crop, t_site_to_t, rot_allo2ego
from utils.weight_init import normal_init, constant_init


class ConvPnPNet(nn.Module):
    def __init__(self, nIn, featdim=128, rot_dim=6, num_layers=3, num_gn_groups=32):
        super().__init__()

        assert num_layers >= 3, num_layers
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
            K = torch.tensor([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899]]) / torch.tensor([[640.], [480.]])
            w0 = torch.eye(5, 6)
            w0[3:, 3:] = K
            weight = pnp_params['pnp_net.features.0.weight'][:, :5]  # [o, 5, k, k]
            if gdr_mode and False:
                self.gdr_conv = nn.Conv2d(5, 5, 1)
                self.gdr_conv.weight = nn.Parameter(w0[:, :-1, None, None])
                self.gdr_conv.bias = nn.Parameter(w0[:, -1])
                self.conv_layers[0].weight = nn.Parameter(weight)
                self.gdr_conv = self.gdr_conv.to(weight.device)
            else:
                weight = weight.permute(2, 3, 0, 1) @ w0[None, None].to(weight.device)
                weight = weight.permute(2, 3, 0, 1)  # [o, 6, k, k]
                self.conv_layers[0].weight = nn.Parameter(weight[:, :-1])
                self.conv_layers[0].bias = nn.Parameter(weight[:, -1].sum(dim=(-2, -1)))
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
            pred_coor3d = sample.gt_coord3d
        else:
            pred_coor3d = (pred_coor3d - .5) * sample.obj_size[..., None, None]  # denormalize from [0., 1.]

        x = torch.cat([pred_coor3d, sample.coord2d], dim=1)

        if gdr_mode and False:
            x = self.gdr_conv(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        pred_cam_R_m2c_6d, pred_cam_t_m2c_site = self.fc_R(x), self.fc_t(x)
        pred_cam_R_m2c_allo = rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        if gdr_mode:
            pred_cam_R_m2c_allo = pred_cam_R_m2c_allo.transpose(-2, -1)
        crop_size, *_ = calc_bbox2d_crop(sample.bbox)
        pred_cam_t_m2c = t_site_to_t(pred_cam_t_m2c_site, sample.bbox,
                                     pnp_input_size / crop_size, sample.cam_K)
        pred_cam_R_m2c = rot_allo2ego(pred_cam_t_m2c) @ pred_cam_R_m2c_allo
        return pred_cam_R_m2c, pred_cam_t_m2c, pred_cam_R_m2c_6d, pred_cam_R_m2c_allo, pred_cam_t_m2c_site
