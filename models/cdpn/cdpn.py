from typing import Optional

import pytorch3d.transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample
from models.cdpn.head import Head
from models.eval.loss import Loss
from models.eval.score import Score
from models.resnet_backbone import ResnetBackbone
from models.texture_net_v import TextureNetV
from models.texture_net_p import TextureNetP
from renderer.scene import Scene
import utils.image_2d
import utils.transform_3d


class CDPN(pl.LightningModule):
    def __init__(self, cfg, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.transform = T.Compose([T.ColorJitter(**cfg.augmentation)])
        self.texture_net_v = None  # TextureNet(objects)
        # self.texture_net_p = TextureNetP(in_channels=6, out_channels=3, n_layers=3, hidden_size=128)
        self.rotation_backbone = ResnetBackbone(in_channels=45)
        self.rotation_head = Head(512, num_layers=0, num_filters=512, kernel_size=3, output_dim=6+3)
        self.loss = Loss(objects_eval if objects_eval is not None else objects, geo=False)
        self.score = Score(objects_eval if objects_eval is not None else objects)

    def forward(self, sample: Sample):
        gt_position_info_roi = torch.cat([
            sample.gt_coord_3d_roi,
            sample.gt_normal_roi,
        ], dim=1)
        gt_position_info_roi = torch.cat([sample.gt_mask_vis_roi, sample.coord_2d_roi, gt_position_info_roi] +
            [(gt_position_info_roi * i).sin() for i in [1, 2, 4, 8, 16, 32]], dim=1)
        sample.img_roi = gt_position_info_roi[:, -6:-3]

        # gt_texel_roi = self.texture_net_p(torch.cat([sample.gt_coord_3d_roi, sample.gt_normal_roi], dim=1))
        # sample.img_roi = (sample.gt_light_texel_roi * gt_texel_roi + sample.gt_light_specular_roi).clamp(0., 1.)
        # sample.img_roi = self.transform(sample.img_roi)
        # input_features = torch.cat([sample.img_roi, sample.coord_2d_roi], dim=1)

        # translation_features = self.translation_backbone(input_features)
        # sample.pred_cam_t_m2c_site = self.translation_head(translation_features)

        rotation_features = self.rotation_backbone(gt_position_info_roi)
        pred_cam_R_m2c_6d, sample.pred_cam_t_m2c_site = self.rotation_head(rotation_features).split([6, 3], dim=1)
        pred_cam_R_m2c_allo = pytorch3d.transforms.rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        if self.training:
            rot_allo2ego = utils.transform_3d.rot_allo2ego(sample.gt_cam_t_m2c)
        else:
            rot_allo2ego = utils.transform_3d.rot_allo2ego(sample.pred_cam_t_m2c)
        sample.pred_cam_R_m2c = rot_allo2ego @ pred_cam_R_m2c_allo
        # sample.visualize(max_samples=4)
        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_pm, loss_t_site_center, loss_t_site_depth = self.loss(sample)
        loss_pm, loss_t_site_center, loss_t_site_depth = loss_pm.mean(), loss_t_site_center.mean(), loss_t_site_depth.mean()
        loss = loss_pm + loss_t_site_center + loss_t_site_depth
        self.log('loss', {'total': loss, 'pm': loss_pm, 'ts_center': loss_t_site_center, 'ts_depth': loss_t_site_depth})
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

    def configure_optimizers(self):
        params = [
            {'params': self.rotation_backbone.parameters(), 'lr': 1e-4, 'name': 'rotation_backbone'},
            # {'params': self.translation_backbone.parameters(), 'lr': 1e-4, 'name': 'translation_backbone'},
            {'params': self.rotation_head.parameters(), 'lr': 1e-4, 'name': 'rotation_head'},
            # {'params': self.translation_head.parameters(), 'lr': 1e-4, 'name': 'translation_head'},
            # {'params': self.texture_net_v.parameters(), 'lr': 1e-4, 'name': 'texture_net_v'},
            # {'params': self.texture_net_p.parameters(), 'lr': 1e-4, 'name': 'texture_net_p'},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
        return [optimizer], [scheduler]

    def _log_sample_visualizations(self, sample: Sample) -> None:
        writer: SummaryWriter = self.logger.experiment
        figs = sample.visualize(return_figs=True, max_samples=16)
        count = {}
        for obj_id, fig in zip(sample.obj_id, figs):
            obj_id = int(obj_id)
            c = count[obj_id] if obj_id in count else 0
            writer.add_figure(f'{obj_id}-{self.loss.objects_eval[obj_id].name}-{c}', fig,
                              global_step=self.global_step, close=True)
            count[obj_id] = c + 1

    def on_train_start(self):
        writer: SummaryWriter = self.logger.experiment
        writer.add_text('rotation_backbone', str(self.rotation_backbone), global_step=0)
        # writer.add_text('translation_backbone', str(self.translation_backbone), global_step=0)
        writer.add_text('rotation_head', str(self.rotation_head), global_step=0)
        # writer.add_text('translation_head', str(self.translation_head), global_step=0)
        # writer.add_text('texture_net_v', str(self.texture_net_v), global_step=0)
        # writer.add_text('texture_net_p', str(self.texture_net_p), global_step=0)
        self.on_validation_start()

    def on_validation_start(self):
        Scene.texture_net_v = self.texture_net_v

    def on_test_start(self):
        self.on_validation_start()

    def on_predict_start(self):
        self.on_validation_start()
