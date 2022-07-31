from typing import Optional

import pytorch3d.ops
import pytorch3d.transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torchvision.transforms.functional as vF

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample, SampleFields as sf
from models.eval.loss import Loss
from models.eval.score import Score
from models.resnet_backbone import ResnetBackbone
from models.gdr.conv_pnp_net import ConvPnPNet
from models.gdr.rot_head import RotWithRegionHead
from models.surfemb.siren import Siren
from models.texture_net_p import TextureNetP
from renderer.scene import Scene
import utils.image_2d
import utils.transform_3d


class SurfEmb(pl.LightningModule):
    def __init__(self, cfg, objects: dict[int, ObjMesh], objects_eval: dict[int, ObjMesh] = None):
        super().__init__()
        self.texture_net_v = None
        self.texture_net_p = None  # TextureNetP(in_channels=6+36+36, out_channels=3, n_layers=3, hidden_size=128)
        self.texture_net_e = Siren(in_features=3, out_features=3, hidden_features=256, hidden_layers=2)
        self.objects_eval = objects_eval if objects_eval is not None else objects

    def configure_optimizers(self):
        params = [
            # {'params': self.texture_net_p.parameters(), 'lr': 1e-6, 'name': 'texture_p'},
            {'params': self.texture_net_e.parameters(), 'lr': 3e-4, 'name': 'texture_e'},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
        return [optimizer], [scheduler]

    def forward(self, sample: Sample):
        loss = []
        for i in range(sample.get(sf.N)):
            S_tilde: torch.Tensor = pytorch3d.ops.sample_points_from_meshes(
                self.objects_eval[int(sample.get(sf.obj_id)[i])].mesh,
                num_samples=1024, return_normals=False, return_textures=False)[0]  # [V, 3(XYZ)]
            U = sample.get(sf.gt_mask_vis_roi)[i, 0].nonzero()  # [U, 2(vu)]
            gt_coord_3d_roi = sample.get(sf.gt_coord_3d_roi)[i]
            C_u = torch.stack([gt_coord_3d_roi[:, v, u] for v, u in U[torch.randint(len(U), [1024])]])
            # [U, 3(XYZ)]
            # S_tilde_emb = self.texture_net_p(S_tilde[..., None, None])[..., 0, 0]  # [V, E]
            # C_u_emb = self.texture_net_p(C_u[..., None, None])[..., 0, 0]  # [U, E]
            S_tilde_emb = self.texture_net_e(S_tilde).sin()  # [V, E]
            C_u_emb = self.texture_net_e(C_u).sin()  # [U, E]

            def similarity(q: torch.Tensor, k: torch.Tensor):
                """

                :param q: [..., E], broadcast-able
                :param k: [..., E], broadcast-able
                :return: [...], broadcast dims
                """
                # sim = (q * k).sum(dim=-1).exp()
                sim = (-((q - k) ** 2).sum(dim=-1)).exp()
                return sim

            numerators = similarity(C_u_emb, C_u_emb)
            denominators = similarity(C_u_emb[..., None, :], S_tilde_emb).sum(dim=-1)
            prs = numerators / (numerators + denominators)
            loss.append(-prs.log().mean())

        sample.loss = torch.stack(loss)
        # sample.visualize()
        return sample

    def training_step(self, sample: Sample, batch_idx: int) -> STEP_OUTPUT:
        sample = self.forward(sample)
        loss = sample.loss.sum()
        self.log('loss', {'total': loss})
        return loss

    def validation_step(self, sample: Sample, batch_idx: int) -> Optional[STEP_OUTPUT]:
        gt_coord_3d_roi = sample.get(sf.gt_coord_3d_roi)
        # sample.img_roi = self.texture_net_p(gt_coord_3d_roi)
        N, C, H, W = gt_coord_3d_roi.shape
        gt_coord_3d_roi_linear = gt_coord_3d_roi.permute(0, 2, 3, 1).reshape(-1, C)
        img_roi = self.texture_net_e(gt_coord_3d_roi_linear).sin()
        sample.img_roi = img_roi.reshape(N, H, W, C).permute(0, 3, 1, 2)
        if (batch_idx + 1) % (self.trainer.log_every_n_steps * 999) == 1:
            self._log_sample_visualizations(sample)
        return None
        re, te, add, proj = self.score(sample)
        metric_dict = {'re(deg)': re, 'te(cm)': te, 'ad(d)': add, 'proj': proj}
        return metric_dict

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('val_metric', 1.)
        return None
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
