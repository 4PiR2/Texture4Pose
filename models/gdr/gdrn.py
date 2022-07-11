from typing import Optional

import pytorch3d.transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as vF

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample
from models.eval.loss import Loss
from models.eval.score import Score
from models.resnet_backbone import ResnetBackbone
from models.gdr.conv_pnp_net import ConvPnPNet
from models.gdr.rot_head import RotWithRegionHead
from models.texture_net_p import TextureNetP
from renderer.scene import Scene
import utils.image_2d
import utils.transform_3d


class GDRN(pl.LightningModule):
    def __init__(self, cfg, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.transform = T.Compose([T.ColorJitter(**cfg.augmentation)])
        self.texture_net_v = None  # TextureNetV(objects)
        self.texture_net_p = TextureNetP(in_channels=6+36+36, out_channels=3, n_layers=3, hidden_size=128)
        self.backbone = ResnetBackbone(in_channels=3)
        self.rot_head_net = RotWithRegionHead(512, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1)
        self.pnp_net = ConvPnPNet(in_channels=6)
        self.objects_eval = objects_eval if objects_eval is not None else objects
        self.loss = Loss(objects_eval if objects_eval is not None else objects)
        self.score = Score(objects_eval if objects_eval is not None else objects)

    def configure_optimizers(self):
        params = [
            {'params': self.backbone.parameters(), 'lr': 1e-5, 'name': 'backbone'},
            {'params': self.rot_head_net.parameters(), 'lr': 1e-5, 'name': 'rot_head'},
            {'params': self.pnp_net.parameters(), 'lr': 1e-5, 'name': 'pnp'},
            {'params': self.texture_net_p.parameters(), 'lr': 1e-6, 'name': 'texture_p'},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
        return [optimizer], [scheduler]

    def forward(self, sample: Sample):
        gt_position_info_roi = torch.cat([
            sample.gt_coord_3d_roi_normalized,
            sample.gt_normal_roi,
        ], dim=1)
        gt_position_info_roi = torch.cat([gt_position_info_roi] \
            + [x.sin() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32]]] \
            + [x.cos() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32]]],
        dim=1)
        gt_texel_roi = self.texture_net_p(gt_position_info_roi)
        sample.img_roi = (sample.gt_light_texel_roi * gt_texel_roi + sample.gt_light_specular_roi).clamp(0., 1.)
        sample.img_roi = self.transform(sample.img_roi)
        sample.gt_mask_vis_roi = vF.resize(sample.gt_mask_obj_roi, [64])  # mask: vis changed to obj
        sample.gt_coord_3d_roi = vF.resize(sample.gt_coord_3d_roi, [64]) * sample.gt_mask_vis_roi
        sample.coord_2d_roi = vF.resize(sample.coord_2d_roi, [64])

        features = self.backbone(sample.img_roi)
        pred_mask_vis_roi, sample.pred_coord_3d_roi_normalized = self.rot_head_net(features)
        sample.pred_mask_vis_roi = pred_mask_vis_roi  #.sigmoid()

        if self.training:
            pred_cam_R_m2c_6d, sample.pred_cam_t_m2c_site = self.pnp_net(
                sample.gt_coord_3d_roi, sample.coord_2d_roi, sample.gt_mask_vis_roi
            )
        else:
            pred_cam_R_m2c_6d, sample.pred_cam_t_m2c_site = self.pnp_net(
                sample.pred_coord_3d_roi * (sample.pred_mask_vis_roi > .5),
                sample.coord_2d_roi, (sample.pred_mask_vis_roi > .5).to(dtype=sample.pred_mask_vis_roi.dtype)
            )
        pred_cam_R_m2c_allo = pytorch3d.transforms.rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        if self.training:
            sample.pred_cam_R_m2c = utils.transform_3d.rot_allo2ego(sample.gt_cam_t_m2c) @ pred_cam_R_m2c_allo
        else:
            sample.pred_cam_R_m2c = utils.transform_3d.rot_allo2ego(sample.pred_cam_t_m2c) @ pred_cam_R_m2c_allo
        if not self.training:
            sample.compute_pnp(erode_min=5.)
        # sample.visualize()
        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_pm, loss_t_site_center, loss_t_site_depth, loss_coord_3d, loss_mask = self.loss(sample)
        loss = loss_coord_3d + loss_mask + loss_pm + loss_t_site_center + loss_t_site_depth
        self.log('loss', {'total': loss, 'coord_3d': loss_coord_3d, 'mask': loss_mask, 'pm': loss_pm,
                          'ts_center': loss_t_site_center, 'ts_depth': loss_t_site_depth})
        return loss

    def validation_step(self, sample: Sample, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample: Sample = self.forward(sample)
        if (batch_idx + 1) % (self.trainer.log_every_n_steps * 999) == 1:
            self._log_sample_visualizations(sample)
        re, te, add, proj = self.score(sample)
        metric_dict = {'re(deg)': re, 'te(cm)': te, 'ad(d)': add, 'proj': proj}
        return metric_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        keys = list(outputs[0].keys())
        outputs = {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}
        metrics = torch.stack([outputs[key] for key in keys], dim=0)
        q = torch.linspace(0., 1., 9, dtype=metrics.dtype, device=metrics.device)[1:-1]
        quantiles = metrics.quantile(q, dim=1).T
        for i in range(len(keys)):
            self.log(keys[i], {f'%{int((q[j] * 100.).round())}': float(quantiles[i, j]) for j in range(len(q))})
        self.log('val_metric', metrics[2].mean())  # mean add score, for model selection

    def _log_sample_visualizations(self, sample: Sample) -> None:
        writer: SummaryWriter = self.logger.experiment
        figs = sample.visualize(return_figs=True)
        count = {}
        for obj_id, fig in zip(sample.obj_id, figs):
            obj_id = int(obj_id)
            c = count[obj_id] if obj_id in count else 0
            writer.add_figure(f'{obj_id}-{self.objects_eval[obj_id].name}-{c}', fig,
                              global_step=self.global_step, close=True)
            count[obj_id] = c + 1

    def on_train_start(self):
        writer: SummaryWriter = self.logger.experiment
        for key in dir(self):
            if key.startswith('_'):
                continue
            value = getattr(self, key)
            if isinstance(value, nn.Module):
                writer.add_text(key, str(value), global_step=0)
        self.on_validation_start()

    def on_validation_start(self):
        Scene.texture_net_v = self.texture_net_v

    def on_test_start(self):
        self.on_validation_start()

    def on_predict_start(self):
        self.on_validation_start()
