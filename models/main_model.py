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
from dataloader.sample import Sample, SampleFields as sf
from models.backbone.resnet_backbone import ResnetBackbone
from models.backbone.up_sampling_backbone import UpSamplingBackbone
from models.epropnp.demo import EProPnPDemo
from models.eval.loss import Loss
from models.eval.score import Score
from models.head.epropnp_head import EPHead
from models.head.geo_head import GeoHead
from models.texture_net.siren_conv import SirenConv
from models.texture_net.texture_net_p import TextureNetP
from models.texture_net.texture_net_v import TextureNetV
from renderer.scene import Scene


class MainModel(pl.LightningModule):
    def __init__(self, cfg, objects: dict[int, ObjMesh] = None, objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        # self.cfg = cfg
        self.pnp_input_size: int = cfg.model.pnp_input_size
        self.texture_mode: str = cfg.model.texture_mode
        self.pnp_mode: str = cfg.model.pnp_mode
        self.opt_cfg = cfg.optimizer
        self.eval_augmentation: bool = cfg.model.eval_augmentation
        self.match_background_histogram_cfg = cfg.augmentation.match_background_histogram
        self.epro_use_world_measurement: bool = cfg.model.pnp.epro_use_world_measurement
        self.epro_loss_weight: float = cfg.model.pnp.epro_loss_weight

        self.transform = T.Compose([
            A.CoarseDropout(**cfg.augmentation.coarse_dropout),
            A.Debayer(**cfg.augmentation.debayer),
            A.MotionBlur(**cfg.augmentation.motion_blur),
            A.GaussianBlur(**cfg.augmentation.gaussian_blur),
            A.Sharpen(**cfg.augmentation.sharpen),
            A.ISONoise(**cfg.augmentation.iso_noise),
            A.GaussNoise(**cfg.augmentation.gauss_noise),
            A.ColorJitter(**cfg.augmentation.color_jitter),
            # T.RandomAutocontrast(p=.5),
            # T.RandomEqualize(p=.5),
            # T.Lambda(vF.adjust_gamma),
        ])
        if self.texture_mode == 'vertex':
            self.texture_net_v = TextureNetV(objects)
        else:
            self.texture_net_v = None
            if self.texture_mode == 'mlp':
                self.texture_net_p = TextureNetP(in_channels=3, out_channels=3, n_layers=3, hidden_size=128,
                    positional_encoding=[2. ** i for i in range(8)], use_cosine_positional_encoding=True)
            elif self.texture_mode == 'siren':
                self.texture_net_p = SirenConv(in_features=3, out_features=3, hidden_features=128, hidden_layers=2,
                                               outermost_linear=False, first_omega_0=30., hidden_omega_0=30.)
            else:
                self.texture_net_p = None

        num_hidden = 256
        self.resnet_backbone = ResnetBackbone(in_channels=3)
        self.up_sampling_backbone = UpSamplingBackbone(in_channels=512, num_layers=6, hidden_channels=num_hidden,
                                                       kernel_size=3)
        self.coord_3d_head = GeoHead(in_channels=num_hidden, out_channels=3, kernel_size=1)

        if self.pnp_mode == 'epro':
            self.secondary_head = EPHead(in_channels=num_hidden, kernel_size=1)
            self.pnp_net = EProPnPDemo()
        elif self.pnp_mode == 'gdrn':
            self.secondary_head = GeoHead(in_channels=num_hidden, out_channels=1, kernel_size=1)
            # self.pnp_net =
            raise NotImplementedError
        elif self.pnp_mode == 'ransac':
            self.secondary_head = GeoHead(in_channels=num_hidden, out_channels=1, kernel_size=1)
            self.pnp_net = None
        else:
            self.secondary_head=None
            self.pnp_net = None

        self.objects_eval = objects_eval if objects_eval is not None else objects
        self.loss = Loss(objects_eval if objects_eval is not None else objects)
        self.score = Score(objects_eval if objects_eval is not None else objects)

    def configure_optimizers(self):
        params = [
            {'params': self.resnet_backbone.parameters(), 'lr': self.opt_cfg.lr.resnet_backbone,
             'name': 'resnet_backbone'},
            {'params': self.up_sampling_backbone.parameters(), 'lr': self.opt_cfg.lr.up_sampling_backbone,
             'name': 'up_sampling_backbone'},
            {'params': self.coord_3d_head.parameters(), 'lr': self.opt_cfg.lr.coord_3d_head, 'name': 'coord_3d_head'},
        ]
        if self.texture_mode in ['mlp', 'siren']:
            params.append({'params': self.texture_net_p.parameters(), 'lr': self.opt_cfg.lr.texture_net_p,
                           'name': 'texture_p'})
        elif self.texture_mode == 'vertex':
            params.append({'params': self.texture_net_v.parameters(), 'lr': self.opt_cfg.lr.texture_net_v,
                           'name': 'texture_v'})
        if self.secondary_head is not None:
            params.append({'params': self.secondary_head.parameters(), 'lr': self.opt_cfg.lr.secondary_head,
                           'name': 'secondary_head'})
        if self.pnp_mode == 'gdrn':
            params.append({'params': self.pnp_net.parameters(), 'lr': self.opt_cfg.lr.pnp_net, 'name': 'pnp_net'})
        if self.opt_cfg.mode == 'adam':
            opt = torch.optim.Adam
        elif self.opt_cfg.mode == 'sgd':
            opt = torch.optim.SGD
        else:
            raise NotImplementedError
        optimizer = opt(params, lr=0.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
        return [optimizer], [scheduler]

    def forward_texture(self, sample: Sample = None, texture_mode: str = None, coord_3d_normalized: torch.Tensor = None,
                        normal: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        if sample is not None:
            coord_3d_normalized = sample.gt_coord_3d_roi_normalized
            normal = sample.get(sf.gt_normal_roi)
            mask = sample.get(sf.gt_mask_vis_roi)
        if texture_mode is None:
            texture_mode = self.texture_mode
        texel = None
        if texture_mode is not None:
            if self.texture_net_p is not None:
                N, _, H, W = mask.shape
                input_feature_map = coord_3d_normalized * 2. - 1.
                input_feature = input_feature_map.permute(0, 2, 3, 1)[mask[:, 0]][..., None, None]
                # [NHW, C, 1, 1]
                output_feature_map = torch.ones(N, H, W, 3, dtype=input_feature.dtype, device=input_feature.device)
                output_feature_map[mask[:, 0]] = self.texture_net_p(input_feature)[..., 0, 0]
                texel = output_feature_map.permute(0, 3, 1, 2)
            elif texture_mode == 'xyz':
                texel = coord_3d_normalized
            elif texture_mode == 'cb':
                n_cb_cycle = 4
                texel = (coord_3d_normalized * (n_cb_cycle * 2)).int() % 2
            if sample is not None:
                if texel is not None:
                    sample.set(sf.gt_texel_roi, texel)
                sample.set(sf.img_roi, (sample.get(sf.gt_light_texel_roi) * sample.get(sf.gt_texel_roi) +
                                        sample.get(sf.gt_light_specular_roi)).clamp(0., 1.))
        return texel

    def forward(self, sample: Sample):
        self.forward_texture(sample)

        if self.training or self.eval_augmentation:
            sample.set(sf.img_roi, augmentations.color_augmentation.match_background_histogram(
                sample.get(sf.img_roi), sample.get(sf.gt_mask_vis_roi), **self.match_background_histogram_cfg
            ))
            sample.set(sf.img_roi,
                       self.transform(sample.get(sf.img_roi)))

        sample.set(sf.gt_mask_vis_roi,
                   vF.resize(sample.get(sf.gt_mask_obj_roi), [self.pnp_input_size]))  # mask: vis changed to obj
        sample.set(sf.gt_coord_3d_roi,
                   vF.resize(sample.get(sf.gt_coord_3d_roi), [self.pnp_input_size]) * sample.get(sf.gt_mask_vis_roi))
        sample.set(sf.coord_2d_roi,
                   vF.resize(sample.get(sf.coord_2d_roi), [self.pnp_input_size]))

        features = self.up_sampling_backbone(self.resnet_backbone(sample.get(sf.img_roi)))
        sample.set(sf.pred_coord_3d_roi_normalized, self.coord_3d_head(features))

        if self.pnp_mode == 'epro':
            w2d_raw, log_weight_scale = self.secondary_head(features)
            N, _, H, W = w2d_raw.shape
            if not self.training:
                pred_weight_2d = (w2d_raw.detach().reshape(N, 2, -1).log_softmax(dim=-1).reshape(N, 2, H, W)
                                  + log_weight_scale.detach()[..., None, None]).exp()
                sample.set(sf.pred_weight_2d, pred_weight_2d / pred_weight_2d.max())
            x3d = sample.get(sf.pred_coord_3d_roi).permute(0, 2, 3, 1).reshape(N, -1, 3)
            x2d = sample.get(sf.coord_2d_roi).permute(0, 2, 3, 1).reshape(N, -1, 2)
            if not self.epro_use_world_measurement:
                x2d = (sample.get(sf.cam_K)[:, None]
                       @ torch.cat([x2d, torch.ones_like(x2d[..., :1])], dim=-1)[..., None])[..., :-1, 0]
            w2d = w2d_raw.permute(0, 2, 3, 1).reshape(N, -1, 2)
            if self.training:
                out_pose = torch.cat([sample.get(sf.gt_cam_t_m2c),
                                      pytorch3d.transforms.matrix_to_quaternion(sample.get(sf.gt_cam_R_m2c))], dim=-1)
                pose_opt, *_, ep_loss = self.pnp_net.forward_train(x3d, x2d, w2d, log_weight_scale, out_pose,
                                            None if self.epro_use_world_measurement else sample.get(sf.cam_K))
                sample.tmp_eploss = ep_loss
            else:
                pose_opt = self.pnp_net.forward_test(x3d, x2d, w2d, log_weight_scale,
                               None if self.epro_use_world_measurement else sample.get(sf.cam_K), fast_mode=False)
            sample.set(sf.pred_cam_t_m2c, pose_opt[:, :3])
            sample.set(sf.pred_cam_R_m2c, pytorch3d.transforms.quaternion_to_matrix(pose_opt[:, 3:]))

        elif self.pnp_mode == 'ransac':
            sample.compute_pnp(sanity_check_mode=False, store=True, ransac=True)

        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_dict = {}
        loss_dict['loss_coord_3d'] = self.loss.coord_3d_loss(
            sample.gt_coord_3d_roi_normalized, sample.gt_mask_vis_roi, sample.pred_coord_3d_roi_normalized).mean()

        if self.pnp_mode == 'epro':
            loss_dict['epro'] = sample.tmp_eploss * self.epro_loss_weight
            del sample.tmp_eploss

        loss_dict['total'] = sum(loss_dict.values())
        self.log('loss', loss_dict)
        return loss_dict['total']

    def validation_step(self, sample: Sample, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample: Sample = self.forward(sample)
        if (batch_idx + 1) % (self.trainer.log_every_n_steps * 999) == 1:
            self._log_sample_visualizations(sample)
        if self.pnp_mode is not None:
            re, te, add, proj = self.score(sample)
            metric_dict = {'re(deg)': re, 'te(cm)': te, 'ad(d)': add, 'proj': proj}
            return metric_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if outputs:
            keys = list(outputs[0].keys())
            outputs = {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}
            metrics = torch.stack([outputs[key] for key in keys], dim=0)
            q = torch.linspace(0., 1., 9, dtype=metrics.dtype, device=metrics.device)[1:-1]
            quantiles = metrics.quantile(q, dim=1).T
            for i in range(len(keys)):
                self.log(keys[i], {f'%{int((q[j] * 100.).round())}': float(quantiles[i, j]) for j in range(len(q))})
            self.log('val_metric', metrics[2].mean())  # mean add score, for model selection
        else:
            self.log('val_metric', 1. / (self.current_epoch + 1.))  # mean add score, for model selection

    def _log_sample_visualizations(self, sample: Sample) -> None:
        writer: SummaryWriter = self.logger.experiment
        figs = sample.visualize(return_figs=True, max_samples=16)
        count = {}
        for obj_id, fig in zip(sample.get(sf.obj_id), figs):
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
