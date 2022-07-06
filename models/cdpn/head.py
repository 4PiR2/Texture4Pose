from torch import nn


class Head(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_dim=3, with_bias_end=True):
        super().__init__()

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0

        conv = []
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            conv.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            conv.append(nn.BatchNorm2d(num_filters))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            nn.Linear(num_filters * 8 * 8, 4096),
            nn.LeakyReLU(.1, inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(.1, inplace=True),
            nn.Linear(4096, output_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and (m.bias is not None):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
