from torch import nn
import torchvision


class ResnetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        if in_channels == 3:
            self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=True)
            self.backbone.avgpool = nn.Sequential()
        else:
            self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=False)
            self.backbone.conv1 = nn.Conv2d(in_channels,
                                            self.backbone.conv1.out_channels,
                                            kernel_size=self.backbone.conv1.kernel_size,
                                            stride=self.backbone.conv1.stride, padding=self.backbone.conv1.padding,
                                            bias=self.backbone.conv1.bias is not None)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()

    def forward(self, x):
        x = self.backbone(x)
        size = int((x.shape[-1] // 512) ** .5)
        return x.view(-1, 512, size, size)