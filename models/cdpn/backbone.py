from torch import nn
import torchvision


class ResnetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(pretrained=True)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()

    def forward(self, x):
        return self.backbone.forward(x).view(-1, 512, 8, 8)
