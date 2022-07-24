from torch import nn
import torchvision


class ResnetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        if in_channels == 3:
            self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=True)
        else:
            self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=False)
            self.backbone.conv1 = nn.Conv2d(in_channels,
                                            self.backbone.conv1.out_channels,
                                            kernel_size=self.backbone.conv1.kernel_size,
                                            stride=self.backbone.conv1.stride, padding=self.backbone.conv1.padding,
                                            bias=self.backbone.conv1.bias is not None)
        self.backbone.fc = nn.Sequential()

    def forward(self, x):
        backbone = self.backbone

        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
