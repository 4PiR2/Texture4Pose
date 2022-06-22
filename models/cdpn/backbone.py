from torch import nn
import torchvision


class ResnetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(in_channels,
            self.backbone.conv1.out_channels, kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride, padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias is not None)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()

    def forward(self, x):
        return self.backbone.forward(x).view(-1, 512, 8, 8)
