from typing import Optional

import pytorch3d.transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as vF

import augmentations.color_augmentation
import augmentations.wrapper as A
from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample
from models.epropnp.demo import EProPnPDemo
from models.eval.loss import Loss
from models.eval.score import Score
from models.gdr2.siren_conv import SirenConv
from models.resnet_backbone import ResnetBackbone
from models.gdr2.rot_head import RotWithRegionHead
from models.texture_net_p import TextureNetP
from renderer.scene import Scene
import utils.color
import utils.image_2d
import utils.transform_3d


class GDRN(pl.LightningModule):
    def __init__(self, cfg, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.transform = T.Compose([
            A.CoarseDropout(num_holes=10, width=8, p=.5),
            A.Debayer(permute_channel=True, p=.5),
            A.MotionBlur(kernel_size=(1., 9.), p=.5),
            A.GaussianBlur(sigma=(1., 3.), p=.5),
            A.Sharpen(sharpness_factor=(1., 3.), p=.5),
            A.ISONoise(color_shift=.05, intensity=.1, p=.5),
            A.GaussNoise(sigma=.1, p=.5),
            A.ColorJitter(**cfg.augmentation, p=.5),
            # T.RandomAutocontrast(p=.5),
            # T.RandomEqualize(p=.5),
            # vF.adjust_gamma(),
        ])
        self.texture_net_v = None  # TextureNetV(objects)
        self.texture_net_p = None
        # self.texture_net_p = TextureNetP(in_channels=3*(1+8*2), out_channels=3, n_layers=3, hidden_size=128)
        self.texture_net_p = SirenConv(in_features=3, out_features=3, hidden_features=128, hidden_layers=2, outermost_linear=False)
        self.backbone = ResnetBackbone(in_channels=3)
        self.rot_head_net = RotWithRegionHead(512, out_channels=3+2, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1)
        self.objects_eval = objects_eval if objects_eval is not None else objects
        self.loss = Loss(objects_eval if objects_eval is not None else objects)
        self.score = Score(objects_eval if objects_eval is not None else objects)
        self.epropnp = EProPnPDemo()

    def configure_optimizers(self):
        params = [
            {'params': self.backbone.parameters(), 'lr': 3e-4, 'name': 'backbone'},
            {'params': self.rot_head_net.parameters(), 'lr': 3e-4, 'name': 'rot_head'},
            {'params': self.texture_net_p.parameters(), 'lr': 1e-6, 'name': 'texture_p'},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
        return [optimizer], [scheduler]

    def forward(self, sample: Sample):
        if self.texture_net_p is not None:
            gt_position_info_roi = torch.cat([
                sample.gt_coord_3d_roi_normalized,
                # sample.gt_normal_roi,
            ], dim=1)
            # gt_position_info_roi = torch.cat([gt_position_info_roi] \
            #     + [(x * (torch.pi * 2.)).sin() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32, 64, 128]]] \
            #     + [(x * (torch.pi * 2.)).cos() for x in [gt_position_info_roi * i for i in [1, 2, 4, 8, 16, 32, 64, 128]]],
            # dim=1)
            sample.gt_texel_roi = self.texture_net_p(gt_position_info_roi) *.5 + .5
            sample.img_roi = (sample.gt_light_texel_roi * sample.gt_texel_roi + sample.gt_light_specular_roi).clamp(0., 1.)

        # gt_texel_roi = sample.gt_coord_3d_roi_normalized  # XYZ texture
        # sample.img_roi = (sample.gt_light_texel_roi * gt_texel_roi + sample.gt_light_specular_roi).clamp(0., 1.)
        if self.training or True:
            sample.img_roi = augmentations.color_augmentation.match_background_histogram(
                sample.img_roi, sample.gt_mask_vis_roi, blend_saturation=1., blend_light=1., p=.5
            )
            sample.img_roi = self.transform(sample.img_roi)

        sample.gt_mask_vis_roi = vF.resize(sample.gt_mask_obj_roi, [64])  # mask: vis changed to obj
        sample.gt_coord_3d_roi = vF.resize(sample.gt_coord_3d_roi, [64]) * sample.gt_mask_vis_roi
        sample.coord_2d_roi = vF.resize(sample.coord_2d_roi, [64])

        features = self.backbone(sample.img_roi)
        features, log_weight_scale = self.rot_head_net(features)
        sample.pred_coord_3d_roi_normalized, w2d_raw = features.split([3, 2], dim=-3)
        N = len(sample)
        sample.pred_weight_2d = (w2d_raw.detach().reshape(N, 2, -1).log_softmax(dim=-1).reshape(N, 2, 64, 64)
                                 + log_weight_scale.detach()[..., None, None]).exp()
        sample.pred_weight_2d /= sample.pred_weight_2d.max()
        x3d = sample.pred_coord_3d_roi.permute(0, 2, 3, 1).reshape(N, -1, 3)
        x2d = sample.coord_2d_roi.permute(0, 2, 3, 1).reshape(N, -1, 2)
        w2d = w2d_raw.permute(0, 2, 3, 1).reshape(N, -1, 2)

        if self.training:
            out_pose = torch.cat(
                [sample.gt_cam_t_m2c, pytorch3d.transforms.matrix_to_quaternion(sample.gt_cam_R_m2c)], dim=-1)
            pose_opt, *_, loss = self.epropnp.forward_train(x3d, x2d, w2d, log_weight_scale, out_pose)
            sample.loss = loss
        else:
            pose_opt = self.epropnp.forward_test(x3d, x2d, w2d, log_weight_scale, fast_mode=False)
        t_site = utils.transform_3d.t_to_t_site(pose_opt[:, :3], sample.bbox, sample.roi_zoom_in_ratio, sample.cam_K)
        sample.pred_cam_t_m2c_site = t_site
        sample.pred_cam_R_m2c = pytorch3d.transforms.quaternion_to_matrix(pose_opt[:, 3:])

        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_coord_3d = self.loss.coord_3d_loss(
            sample.gt_coord_3d_roi_normalized, sample.gt_mask_vis_roi, sample.pred_coord_3d_roi_normalized).mean()
        loss_epro = sample.loss * .02
        loss = loss_coord_3d + loss_epro
        self.log('loss', {'total': loss, 'coord_3d': loss_coord_3d, 'loss_epro': loss_epro})
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
        Scene.texture_net_p = self.texture_net_p

    def on_test_start(self):
        self.on_validation_start()

    def on_predict_start(self):
        self.on_validation_start()
