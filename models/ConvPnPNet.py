import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init


class ConvPnPNet(nn.Module):
    def __init__(self, nIn, featdim=128, rot_dim=6, num_layers=3, num_gn_groups=32):
        """
        Args:
            nIn: input feature channel
        """
        super().__init__()

        assert num_layers >= 3, num_layers
        features = []
        for i in range(3):
            _in_channels = nIn if i == 0 else featdim
            stride = 2 if i < 3 else 1
            features.append(nn.Conv2d(_in_channels, featdim, kernel_size=3, stride=stride, padding=1, bias=False))
            features.append(nn.GroupNorm(num_gn_groups, featdim))
            features.append(nn.ReLU(inplace=True))

        features.append(nn.Flatten())
        features.append(nn.Linear(featdim * 8 * 8, 1024))
        features.append(nn.LeakyReLU(0.1, inplace=True))
        features.append(nn.Linear(1024, 256))
        features.append(nn.LeakyReLU(0.1, inplace=True))
        self.features = nn.Sequential(*features)

        self.fc_out = nn.Linear(256, rot_dim + 3)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_out, std=0.01)

    def forward(self, x):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """

        x = self.features(x)
        x = self.fc_out(x)

        rot, t = x.split([x.shape[-1] - 3, 3], dim=1)
        return rot, t
