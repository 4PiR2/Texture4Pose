import os
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
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
from models.head.conv_pnp_net import ConvPnPNet
from models.head.epropnp_head import EPHead
from models.head.geo_head import GeoHead
from models.texture_net.siren_conv import SirenConv
from models.texture_net.texture_net_p import TextureNetP
from models.texture_net.texture_net_v import TextureNetV
from renderer.scene import Scene
from utils.config import Config
import utils.transform_3d


class MainModel(pl.LightningModule):
    def __init__(self, cfg: Config, objects: dict[int, ObjMesh] = None, objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.cfg: Config = cfg
        self.pnp_input_size: int = cfg.model.pnp_input_size
        self.texture_mode: str = cfg.model.texture_mode
        self.freeze_texture_net_p: bool = cfg.model.freeze_texture_net_p
        self.freeze_resnet_backbone: bool = cfg.model.freeze_resnet_backbone
        self.pnp_mode: str = cfg.model.pnp_mode
        self.opt_cfg: Config = cfg.optimizer
        self.sch_cfg: Config = cfg.scheduler
        self.eval_augmentation: bool = cfg.model.eval_augmentation
        self.match_background_histogram_cfg: Config = cfg.augmentation.match_background_histogram
        self.texture_use_normal_input: bool = cfg.model.texture.texture_use_normal_input
        self.coord_3d_loss_weights: list[float] = cfg.model.coord_3d_loss_weights
        self.coord_3d_loss_weight_step: int = cfg.model.coord_3d_loss_weight_step
        if self.texture_mode in ['siren', 'scb']:
            self.siren_first_omega_0: float = cfg.model.texture.siren_first_omega_0
            self.siren_hidden_omega_0: float = cfg.model.texture.siren_hidden_omega_0
        if self.texture_mode in ['cb', 'scb']:
            self.cb_num_cycles: int = cfg.model.texture.cb_num_cycles
        if self.pnp_mode == 'epro':
            self.epro_use_world_measurement: bool = cfg.model.pnp.epro_use_world_measurement
            self.epro_loss_weights: list[float] = cfg.model.pnp.epro_loss_weights
            self.epro_loss_weight_step: int = cfg.model.pnp.epro_loss_weight_step
        if self.pnp_mode == 'gdrn':
            self.gdrn_teacher_force: bool = cfg.model.pnp.gdrn_teacher_force
            self.gdrn_run_ransac_baseline: bool = cfg.model.pnp.gdrn_run_ransac_baseline
            self.gdrn_pnp_pretrain: bool = cfg.model.pnp.gdrn_pnp_pretrain

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
            elif self.texture_mode in ['siren', 'scb']:
                self.texture_net_p = SirenConv(in_features=3 + self.texture_use_normal_input * 3, out_features=3,
                                               hidden_features=128, hidden_layers=2, outermost_linear=False,
                                               first_omega_0=self.siren_first_omega_0,
                                               hidden_omega_0=self.siren_hidden_omega_0)
            else:
                self.texture_net_p = None

        num_hidden = cfg.model.up_sampling.num_hidden
        self.resnet_backbone = ResnetBackbone(in_channels=3)
        self.up_sampling_backbone = UpSamplingBackbone(in_channels=512, num_layers=6, hidden_channels=num_hidden,
                                                       kernel_size=3)
        self.coord_3d_head = GeoHead(in_channels=num_hidden, out_channels=3, kernel_size=1)

        if self.pnp_mode == 'epro':
            self.secondary_head = EPHead(in_channels=num_hidden, kernel_size=1)
            self.pnp_net = EProPnPDemo()
        if self.pnp_mode == 'ransac':
            self.secondary_head = GeoHead(in_channels=num_hidden, out_channels=1, kernel_size=1)
        if self.pnp_mode == 'gdrn':
            self.pnp_net = ConvPnPNet(in_channels=3+2, featdim=128, num_layers=3, num_gn_groups=32)

        self.objects_eval = objects_eval if objects_eval is not None else objects
        self.loss = Loss(objects_eval if objects_eval is not None else objects)
        self.score = Score(objects_eval if objects_eval is not None else objects)

    def configure_optimizers(self):
        params = []
        if not self.freeze_resnet_backbone:
            params.append(
                {'params': self.resnet_backbone.parameters(), 'lr': self.opt_cfg.lr.resnet_backbone,
                 'name': 'resnet_backbone'})
        params.append(
            {'params': self.up_sampling_backbone.parameters(), 'lr': self.opt_cfg.lr.up_sampling_backbone,
             'name': 'up_sampling_backbone'})
        params.append(
            {'params': self.coord_3d_head.parameters(), 'lr': self.opt_cfg.lr.coord_3d_head, 'name': 'coord_3d_head'})

        if self.texture_mode == 'mlp':
            params.append({'params': self.texture_net_p.parameters(), 'lr': self.opt_cfg.lr.texture_net_p,
                           'name': 'texture_p'})
        if self.texture_mode in ['siren', 'scb'] and not self.freeze_texture_net_p:
            params.append({'params': self.texture_net_p.first_layer.parameters(),
                           # 'lr': self.opt_cfg.lr.texture_net_p * (self.siren_hidden_omega_0 / self.siren_first_omega_0),
                           'lr': self.opt_cfg.lr.texture_net_p,
                           'name': 'siren_first'})
            params.append({'params': self.texture_net_p.rest_layers.parameters(), 'lr': self.opt_cfg.lr.texture_net_p,
                           'name': 'siren_rest'})
        if self.texture_mode == 'vertex':
            params.append({'params': self.texture_net_v.parameters(), 'lr': self.opt_cfg.lr.texture_net_v,
                           'name': 'texture_v'})

        if self.pnp_mode in ['epro', 'ransac']:
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

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.sch_cfg.step_size,
                                                    gamma=self.sch_cfg.gamma)
        return [optimizer], [scheduler]

    def forward_texture(self, sample: Sample = None, texture_mode: str = None, coord_3d_normalized: torch.Tensor = None,
                        normal: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        if sample is not None:
            coord_3d_normalized = sample.get(sf.gt_coord_3d_roi_normalized)
            normal = sample.get(sf.gt_normal_roi)
            mask = sample.get(sf.gt_mask_vis_roi)
        if texture_mode is None:
            texture_mode = self.texture_mode
        texel = None
        if texture_mode is not None:
            if self.texture_net_p is not None:
                N, _, H, W = mask.shape
                input_feature_map = coord_3d_normalized * 2. - 1.
                if self.texture_use_normal_input:
                    input_feature_map = torch.cat([input_feature_map, normal], dim=-3)
                input_feature = input_feature_map.permute(0, 2, 3, 1)[mask[:, 0]][..., None, None]
                # [NHW, C, 1, 1]
                output_feature_map = torch.ones(N, H, W, 3, dtype=input_feature.dtype, device=input_feature.device)
                output_feature_map[mask[:, 0]] = self.texture_net_p(input_feature)[..., 0, 0]
                texel = output_feature_map.permute(0, 3, 1, 2)

                fig = plt.figure(figsize=(10, 10))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                fig.add_axes(ax)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_axis_off()
                fig.patch.set_alpha(0.)
                ax.patch.set_alpha(0.)
                t_np = (texel[0].permute(1, 2, 0) * 255.).round().detach().cpu().numpy().astype('uint8')
                mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
                ax.imshow(np.concatenate([t_np, mask_np], axis=-1))
                # plt.savefig('plots/texel.png')
                # plt.show()

            if texture_mode == 'xyz':
                texel = coord_3d_normalized
            if texture_mode in ['cb', 'scb']:
                checkerboard = (coord_3d_normalized * (self.cb_num_cycles * 2)).int() % 2
                if texture_mode == 'cb':
                    texel = checkerboard
                else:
                    # checkerboard_bw = checkerboard.sum(dim=-3, keepdim=True) % 2
                    # texel = checkerboard * texel + (1. - checkerboard) * (1. - texel)
                    # texel = ((texel + checkerboard) * .5).clamp(min=0., max=1.)
                    # texel = (texel + checkerboard_bw * .5) % 1.
                    texel = checkerboard * texel
                    # texel = checkerboard_bw * texel
            if sample is not None:
                if texel is not None:
                    texel = (texel + ~sample.get(sf.gt_mask_vis_roi)).clamp(min=0., max=1.)  # set background to white
                    sample.set(sf.gt_texel_roi, texel)
                if hasattr(sample, sf.gt_light_texel_roi) and hasattr(sample, sf.gt_light_specular_roi):
                    sample.set(sf.img_roi, (sample.get(sf.gt_light_texel_roi) * sample.get(sf.gt_texel_roi) +
                                            sample.get(sf.gt_light_specular_roi)).clamp(0., 1.))
        return texel

    def forward(self, sample: Sample):
        # sample.visualize()

        if not (hasattr(self, 'gdrn_pnp_pretrain') and self.gdrn_pnp_pretrain):
            sample.get_gt_coord_3d_roi_normalized()

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            from utils.transform_3d import show_pose_mesh
            show_pose_mesh(ax, sample.cam_K[0], sample.gt_cam_R_m2c[0], sample.gt_cam_t_m2c[0], None, sample.bbox[0])
            plt.savefig('plots/mesh/104_2bw.svg')
            plt.show()

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            coord_3d_np = (sample.gt_coord_3d_roi_normalized[0].permute(1, 2, 0) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([coord_3d_np, mask_np], axis=-1))
            # plt.savefig('plots/3d.png')
            # plt.show()

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            coord_3d_np = ((sample.gt_normal_roi[0].permute(1, 2, 0) * .5 + .5) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([coord_3d_np, mask_np], axis=-1))
            # plt.savefig('plots/n.png')
            # plt.show()

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            from utils.image_2d import normalize_channel
            coord_2d_np = ((normalize_channel(sample.coord_2d_roi)[0].permute(1, 2, 0)) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([coord_2d_np, np.zeros_like(mask_np), np.full_like(mask_np, 255)], axis=-1))
            # plt.savefig('plots/2d.png')
            # plt.show()

            if self.texture_net_p is not None and self.freeze_texture_net_p:
                self.texture_net_p.eval()
                with torch.no_grad():
                    self.forward_texture(sample)
            else:
                self.forward_texture(sample)

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            img_np = (sample.img_roi[0].permute(1, 2, 0) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([img_np, mask_np], axis=-1))
            # plt.savefig('plots/blend.png')
            # plt.show()

            # if self.training or self.eval_augmentation:
            #     sample.set(sf.img_roi, augmentations.color_augmentation.match_background_histogram(
            #         sample.get(sf.img_roi), sample.get(sf.gt_mask_vis_roi), **self.match_background_histogram_cfg
            #     ))
            #     sample.set(sf.img_roi, self.transform(sample.get(sf.img_roi)))

            sample.img_roi = vF.adjust_hue(sample.img_roi, .3)

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            img_np = (sample.img_roi[0].permute(1, 2, 0) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([img_np, np.full_like(mask_np, 255)], axis=-1))
            # plt.savefig('plots/aug.png')
            # plt.show()

            resize = lambda tag: vF.resize(sample.get(tag), [self.pnp_input_size], vF.InterpolationMode.NEAREST)
            sample.set(sf.gt_mask_vis_roi, resize(sf.gt_mask_vis_roi))
            sample.set(sf.gt_mask_obj_roi, resize(sf.gt_mask_obj_roi))
            sample.set(sf.gt_coord_3d_roi, resize(sf.gt_coord_3d_roi) * sample.get(sf.gt_mask_vis_roi))
            sample.get_gt_coord_3d_roi_normalized()
            sample.set(sf.gt_normal_roi, resize(sf.gt_normal_roi) * sample.get(sf.gt_mask_vis_roi))
            sample.set(sf.coord_2d_roi, resize(sf.coord_2d_roi))

            if self.freeze_resnet_backbone:
                self.resnet_backbone.eval()
                with torch.no_grad():
                    features = self.resnet_backbone(sample.get(sf.img_roi))
            else:
                features = self.resnet_backbone(sample.get(sf.gt_texel_roi))
            features = self.up_sampling_backbone(features)
            sample.set(sf.pred_coord_3d_roi_normalized, self.coord_3d_head(features))
            sample.get_pred_coord_3d_roi()

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            x3d_np = (sample.pred_coord_3d_roi_normalized[0].permute(1, 2, 0) * 255.).round().clamp(0., 255.).detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([x3d_np, np.full_like(mask_np, 255)], axis=-1))
            # plt.savefig('plots/x3d.png')
            # plt.show()

        if self.pnp_mode == 'epro':
            w2d_raw, log_weight_scale = self.secondary_head(features)
            N, _, H, W = w2d_raw.shape
            if not self.training:
                pred_weight_2d = (w2d_raw.detach().reshape(N, 2, -1).log_softmax(dim=-1).reshape(N, 2, H, W)
                                  + log_weight_scale.detach()[..., None, None]).exp()
                sample.set(sf.pred_weight_2d, pred_weight_2d / pred_weight_2d.max())

            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_axis_off()
            fig.patch.set_alpha(0.)
            ax.patch.set_alpha(0.)
            w2d_np = (sample.pred_weight_2d[0].permute(1, 2, 0) * 255.).round().detach().cpu().numpy().astype('uint8')
            mask_np = (sample.gt_mask_vis_roi[0].permute(1, 2, 0).int() * 255).detach().cpu().numpy().astype('uint8')
            ax.imshow(np.concatenate([w2d_np, np.zeros_like(mask_np), np.full_like(mask_np, 255)], axis=-1))
            # plt.savefig('plots/w2d.png')
            # plt.show()

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

        elif self.pnp_mode in ['gdrn', 'ransac']:
            sample.set(sf.pred_mask_vis_roi, self.secondary_head(features))
            if self.pnp_mode == 'gdrn':
                sample.get_gt_cam_t_m2c_site()
                if self.gdrn_teacher_force or self.gdrn_pnp_pretrain:
                    if self.training or self.gdrn_pnp_pretrain:
                        coord_3d_roi = sample.get(sf.gt_coord_3d_roi)
                    else:
                        coord_3d_roi = sample.get(sf.pred_coord_3d_roi) * (sample.get(sf.pred_mask_vis_roi) > .5)
                else:
                    coord_3d_roi = sample.get(sf.pred_coord_3d_roi)
                pred_cam_R_m2c_6d, pred_cam_t_m2c_site = self.pnp_net(
                    torch.cat([coord_3d_roi, sample.get(sf.coord_2d_roi)], dim=-3))
                sample.set(sf.pred_cam_t_m2c_site, pred_cam_t_m2c_site)
                sample.get_pred_cam_t_m2c()
                pred_cam_R_m2c_allo = pytorch3d.transforms.rotation_6d_to_matrix(pred_cam_R_m2c_6d)
                if self.training and self.gdrn_teacher_force:
                    allo2ego = utils.transform_3d.rot_allo2ego(sample.get(sf.gt_cam_t_m2c))
                else:
                    allo2ego = utils.transform_3d.rot_allo2ego(sample.get(sf.pred_cam_t_m2c))
                sample.set(sf.pred_cam_R_m2c, allo2ego @ pred_cam_R_m2c_allo)
            if self.pnp_mode == 'ransac' or self.gdrn_run_ransac_baseline:
                sample.compute_pnp(sanity_check_mode=False, store=True, ransac=True, erode_min=5., erode_max=torch.inf)

        elif self.pnp_mode == 'sanity':
            sample.compute_pnp(sanity_check_mode=True, store=True, ransac=True)

        # sample.visualize()
        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_dict = {}

        coord_3d_loss_weight = self.coord_3d_loss_weights[min(self.current_epoch // self.coord_3d_loss_weight_step,
                                                          len(self.coord_3d_loss_weights) - 1)]
        loss_dict['coord_3d'] = self.loss.coord_3d_loss(sample.get(sf.gt_coord_3d_roi_normalized),
            sample.get(sf.gt_mask_vis_roi), sample.get(sf.pred_coord_3d_roi_normalized)).mean() * coord_3d_loss_weight

        if self.pnp_mode == 'epro':
            epro_loss_weight = self.epro_loss_weights[min(self.current_epoch // self.epro_loss_weight_step,
                                                          len(self.epro_loss_weights) - 1)]
            loss_dict['epro'] = sample.tmp_eploss * epro_loss_weight
            del sample.tmp_eploss

        loss_dict['total'] = sum(loss_dict.values())
        self.log('loss', loss_dict)
        return loss_dict['total']

    def validation_step(self, sample: Sample, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample: Sample = self.forward(sample)
        # if (batch_idx + 1) % (self.trainer.log_every_n_steps * 999) == 1:
        if not batch_idx:
            self._log_sample_visualizations(sample)
        if self.pnp_mode is not None:
            re, te, ad, pj, pv = self.score(sample)
            metric_dict = {'re(deg)': re, 'te(cm)': te, 'ad(d)': ad, 'pj(px)': pj, 'pv(px)': pv}
        else:
            loss_coord_3d = self.loss.coord_3d_loss(sample.get(sf.gt_coord_3d_roi_normalized),
                                sample.get(sf.gt_mask_vis_roi), sample.get(sf.pred_coord_3d_roi_normalized))
            metric_dict = {'loss3d': loss_coord_3d}
        return metric_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metrics_path = os.path.join(self.trainer.log_dir, f'metrics')
        os.makedirs(metrics_path, exist_ok=True)
        torch.save(outputs, os.path.join(metrics_path, f'epoch={self.current_epoch}-step={self.global_step}.pth'))
        if outputs:
            keys = list(outputs[0].keys())
            outputs = {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}
            metrics = torch.stack([outputs[key] for key in keys], dim=0)
            q = torch.linspace(0., 1., 9, dtype=metrics.dtype, device=metrics.device)[1:-1]
            quantiles = metrics.quantile(q, dim=1).T
            for i in range(len(keys)):
                self.log(keys[i], {f'%{int((q[j] * 100.).round())}': float(quantiles[i, j]) for j in range(len(q))})
            self.log('val_metric', metrics[-1].mean())  # mean add score, for model selection
        else:
            self.log('val_metric', 1. / (self.current_epoch + 1.))  # mean add score, for model selection
        additional_ckpt_component_names = ['texture_net_v', 'texture_net_p']
        for component_name in additional_ckpt_component_names:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if component is None:
                    continue
                component_path = os.path.join(self.trainer.log_dir, f'checkpoints_{component_name}')
                os.makedirs(component_path, exist_ok=True)
                torch.save(self.texture_net_p, os.path.join(component_path,
                                                            f'epoch={self.current_epoch}-step={self.global_step}.ckpt'))

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
        writer.add_text('cfg', self.cfg.pretty_text, global_step=0)
        self.cfg.dump(os.path.join(self.trainer.log_dir, 'cfg.txt'), pretty_text=True)
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
