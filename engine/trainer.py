import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import pytorch_lightning as pl
from torchvision.models import resnet34

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample
from dataloader.scene import Scene
from models.conv_pnp_net import ConvPnPNet
from models.texture_net import TextureNet
from models.rot_head import RotWithRegionHead
from utils.transform_3d import denormalize_coord_3d


class LitModel(pl.LightningModule):
    def __init__(self, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.texture_net = TextureNet(objects)
        self.backbone = resnet34()
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()
        self.rot_head_net = RotWithRegionHead(512, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                                     num_regions=64)
        self.pnp_net = ConvPnPNet(nIn=5)
        self.objects_eval = objects_eval if objects_eval is not None else objects

    def forward(self, sample: Sample):
        # in lightning, forward defines the prediction/inference actions
        features = self.backbone(sample.img_roi)
        features = features.view(-1, 512, 8, 8)
        mask, coord_3d_normalized, _region = self.rot_head_net(features)
        coord_3d = denormalize_coord_3d(coord_3d_normalized, sample.obj_size)
        pred_cam_R_m2c, pred_cam_t_m2c, *other_outputs = self.pnp_net(sample, coord_3d)
        angle = sample.relative_angle(pred_cam_R_m2c, degree=True)
        dist = sample.relative_dist(pred_cam_t_m2c, cm=True)
        add = sample.add_score(self.objects_eval, pred_cam_R_m2c, pred_cam_t_m2c)
        proj_dist = sample.proj_dist(self.objects_eval, pred_cam_R_m2c, pred_cam_t_m2c)
        return pred_cam_R_m2c, pred_cam_t_m2c, [angle, dist, add, proj_dist], *other_outputs

    def training_step(self, sample: Sample, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        features = self.backbone(sample.img_roi)
        features = features.view(-1, 512, 8, 8)
        mask, coord_3d_normalized, _region = self.rot_head_net(features)
        coord_3d = denormalize_coord_3d(coord_3d_normalized, sample.obj_size)
        pred_cam_R_m2c, pred_cam_t_m2c, *_, pred_cam_t_m2c_site = self.pnp_net(sample, coord_3d)
        loss_coord_3d = sample.coord_3d_loss(coord_3d_normalized)
        loss_mask = sample.mask_loss(mask)
        loss_pm = sample.pm_loss(self.objects_eval, pred_cam_R_m2c)
        loss_t_site_center = sample.t_site_center_loss(pred_cam_t_m2c_site)
        loss_t_site_depth = sample.t_site_depth_loss(pred_cam_t_m2c_site)
        total_loss = loss_coord_3d + loss_mask + loss_pm + loss_t_site_center + loss_t_site_depth
        loss = total_loss.mean()
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
