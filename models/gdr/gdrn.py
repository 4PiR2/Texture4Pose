from typing import Optional

from pytorch3d.renderer.mesh import TexturesBase, TexturesVertex
import pytorch3d.transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample
from models.gdr.conv_pnp_net import ConvPnPNet
from models.gdr.rot_head import RotWithRegionHead
from models.texture_net import TextureNet
from renderer.scene import Scene
import utils.image_2d
import utils.transform_3d


class GDRN(pl.LightningModule):
    def __init__(self, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.texture_net = TextureNet(objects)
        self.backbone: torchvision.models.ResNet = torchvision.models.resnet34(num_classes=6+3)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()
        self.rot_head_net = RotWithRegionHead(512, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1)
        self.pnp_net = ConvPnPNet(in_channels=6)
        self.objects_eval = objects_eval if objects_eval is not None else objects

    def forward(self, sample: Sample):
        features = self.backbone(sample.img_roi)
        features = features.view(-1, 512, 8, 8)
        pred_mask_vis_roi, sample.pred_coord_3d_roi_normalized = self.rot_head_net(features)
        sample.pred_mask_vis_roi = pred_mask_vis_roi.sigmoid()
        sample.get_pred_coord_3d_roi(store=True)
        pred_cam_R_m2c_6d, sample.pred_cam_t_m2c_site = self.pnp_net(
            sample.pred_coord_3d_roi * sample.pred_mask_vis_roi.round(), sample.coord_2d_roi, sample.pred_mask_vis_roi.round())
        pred_cam_R_m2c_allo = pytorch3d.transforms.rotation_6d_to_matrix(pred_cam_R_m2c_6d)
        pred_cam_R_m2c_allo = pred_cam_R_m2c_allo.transpose(-2, -1)  # use GDR's pre-trained weights
        sample.pred_cam_t_m2c = sample.get_pred_cam_t_m2c(store=True)
        sample.pred_cam_R_m2c = utils.transform_3d.rot_allo2ego(sample.pred_cam_t_m2c) @ pred_cam_R_m2c_allo
        # coord_3d_roi = sample.pred_coord_3d_roi / torch.linalg.vector_norm(sample.pred_coord_3d_roi, dim=1)[:, None] * (.1 * .5)
        sample.compute_pnp(erode_min=5.)
        # sample.visualize()
        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss_coord_3d = sample.coord_3d_loss().mean()
        loss_mask = sample.mask_loss().mean()
        loss_pm = sample.pm_loss(self.objects_eval, div_diameter=False).mean()
        loss_t_site_center = sample.t_site_center_loss().mean()
        loss_t_site_depth = sample.t_site_depth_loss().mean()
        loss = loss_coord_3d + loss_mask + loss_pm + loss_t_site_center + loss_t_site_depth
        self.log('loss', {'total': loss, 'coord_3d': loss_coord_3d, 'mask': loss_mask, 'pm': loss_pm,
                          'ts_center': loss_t_site_center, 'ts_depth': loss_t_site_depth})
        return loss

    def validation_step(self, sample: Sample, batch_idx: int) -> Optional[STEP_OUTPUT]:
        sample: Sample = self.forward(sample)
        if (batch_idx + 1) % (self.trainer.log_every_n_steps * 999) == 1:
            self._log_sample_visualizations(sample)
        re = sample.relative_angle(degree=True)
        te = sample.relative_dist(cm=True)
        add = sample.add_score(self.objects_eval, div_diameter=True)
        proj = sample.proj_dist(self.objects_eval)
        metric_dict = {'re(deg)': re, 'te(cm)': te, 'ad(d)': add, 'proj': proj}
        if sample.pnp_cam_R_m2c is not None and sample.pnp_cam_t_m2c is not None:
            pnp_re = sample.relative_angle(sample.pnp_cam_R_m2c, degree=True)
            pnp_te = sample.relative_dist(sample.pnp_cam_t_m2c, cm=True)
            pnp_add = sample.add_score(self.objects_eval, sample.pnp_cam_R_m2c, sample.pnp_cam_t_m2c, div_diameter=True)
            pnp_proj = sample.proj_dist(self.objects_eval, sample.pnp_cam_R_m2c, sample.pnp_cam_t_m2c)
            metric_dict.update(
                {'pnp_re(deg)': pnp_re, 'pnp_te(cm)': pnp_te, 'pnp_ad(d)': pnp_add, 'pnp_proj': pnp_proj})
        return metric_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # self._log_meshes()
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
            {'params': self.backbone.parameters(), 'lr': 1e-5, 'name': 'backbone'},
            {'params': self.rot_head_net.parameters(), 'lr': 1e-5, 'name': 'rot_head'},
            {'params': self.pnp_net.parameters(), 'lr': 1e-5, 'name': 'pnp'},
            {'params': self.texture_net.parameters(), 'lr': 1e-3, 'name': 'texture'},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.1)
        return [optimizer], [scheduler]

    def load_pretrain(self, gdr_pth_path: str):
        state_dict = torch.load(gdr_pth_path)['model']
        state_dict['rot_head_net.features.23.weight'] = state_dict['rot_head_net.features.23.weight'][:4]
        state_dict['rot_head_net.features.23.bias'] = state_dict['rot_head_net.features.23.bias'][:4]
        self.load_state_dict(state_dict, strict=False)
        self.backbone.conv1.weight = nn.Parameter(self.backbone.conv1.weight.flip(dims=[1]))
        self.pnp_net.load_pretrain(gdr_pth_path)

    def _log_meshes(self) -> None:
        for obj_id in self.objects_eval:
            obj = self.objects_eval[obj_id]
            verts = obj.mesh.verts_packed()  # [V, 3(XYZ)]
            faces = obj.mesh.faces_packed()  # [F, 3]
            texture: TexturesBase = self.texture_net(obj)
            if isinstance(texture, TexturesVertex):
                v_texture = texture.verts_features_packed()
            else:
                fv_texture = texture.faces_verts_textures_packed()  # [F, 3, 3(RGB)]
                v_texture = torch.empty_like(verts)  # [V, 3(RGB)]
                for i in range(len(verts)):
                    v_texture[i] = fv_texture[faces == i].mean(dim=0)
            v_texture = (v_texture * 255.).to(dtype=torch.uint8)
            config_dict = {'lights': [{'cls': 'AmbientLight', 'color': 0xffffff, 'intensity': 1.}]}
            # ref: https://www.tensorflow.org/graphics/tensorboard#scene_configuration
            writer: SummaryWriter = self.logger.experiment
            writer.add_mesh(f'{obj_id}-{obj.name}', vertices=verts[None], colors=v_texture[None], faces=faces[None],
                            config_dict=config_dict, global_step=self.global_step)

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
        writer.add_text('backbone', str(self.backbone), global_step=0)
        writer.add_text('rot_head_net', str(self.rot_head_net), global_step=0)
        writer.add_text('pnp_net', str(self.pnp_net), global_step=0)
        writer.add_text('texture_net', str(self.texture_net), global_step=0)
        self.on_validation_start()

    def on_validation_start(self):
        Scene.texture_net = self.texture_net

    def on_test_start(self):
        self.on_validation_start()

    def on_predict_start(self):
        self.on_validation_start()
