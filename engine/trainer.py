import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import pytorch_lightning as pl
from torchvision.models import resnet34

from dataloader.ObjMesh import ObjMesh
from dataloader.Sample import Sample
from dataloader.Scene import Scene
from models.ConvPnPNet import ConvPnPNet
from models.TextureNet import TextureNet
from models.rot_head import RotWithRegionHead


class LitModel(pl.LightningModule):
    def __init__(self, objects: dict[int, ObjMesh]):
        super().__init__()
        self.texture_net = TextureNet(objects)
        self.backbone = resnet34()
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()
        self.rot_head_net = RotWithRegionHead(512, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                                     num_regions=64)
        self.pnp_net = ConvPnPNet(nIn=5)

    def forward(self, sample: Sample):
        # in lightning, forward defines the prediction/inference actions
        features = self.backbone(sample.img)
        mask, coor_3d, _region = self.rot_head_net(features)
        pred_cam_R_m2c, pred_cam_t_m2c, *other_outputs = self.pnp_net(sample, coor_3d)
        return pred_cam_R_m2c, pred_cam_t_m2c, *other_outputs

    def training_step(self, sample: Sample, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        features = self.backbone(sample.img)
        features = features.view(-1, 512, 8, 8)
        mask, coord_3d, _region = self.rot_head_net(features)
        # gt = sample.gt_coord3d / sample.obj_size[..., None, None] + .5
        loss_3d = F.l1_loss(coord_3d * sample.gt_mask_vis, sample.gt_coord_3d * sample.gt_mask_vis)
        loss_m = F.l1_loss(mask, sample.gt_mask_vis.float())
        loss = loss_3d + loss_m
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def load_pretrain(self, gdr_pth_path):
        state_dict = torch.load(gdr_pth_path)['model']
        self.load_state_dict(state_dict, strict=False)
        self.backbone.conv1.weight = nn.Parameter(self.backbone.conv1.weight.flip(dims=[1]))
        self.pnp_net.load_pretrain(gdr_pth_path)

    def on_train_start(self):
        # validation/test/predict
        Scene.texture_net = self.texture_net
