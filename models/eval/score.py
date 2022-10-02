import torch
from torch import nn

from dataloader.obj_mesh import ObjMesh
from dataloader.sample import Sample, SampleFields as sf


class Score(nn.Module):
    def __init__(self, objects_eval: dict[int, ObjMesh]):
        super().__init__()
        self.objects_eval: dict[int, ObjMesh] = objects_eval

    def forward(self, sample: Sample, pred_cam_R_m2c: torch.Tensor = None, pred_cam_t_m2c: torch.Tensor = None):
        obj_id = sample.get(sf.obj_id)
        cam_K = sample.get(sf.cam_K)
        gt_cam_R_m2c = sample.get(sf.gt_cam_R_m2c)
        gt_cam_t_m2c = sample.get(sf.gt_cam_t_m2c)

        if pred_cam_R_m2c is None:
            pred_cam_R_m2c = sample.get(sf.pred_cam_R_m2c)
        if pred_cam_t_m2c is None:
            pred_cam_t_m2c = sample.get(sf.pred_cam_t_m2c)

        re = self.relative_angle(gt_cam_R_m2c, pred_cam_R_m2c, degree=True)
        te = self.relative_dist(gt_cam_t_m2c, pred_cam_t_m2c, cm=True)
        ad = self.add_score(obj_id, gt_cam_R_m2c, gt_cam_t_m2c, pred_cam_R_m2c, pred_cam_t_m2c, div_diameter=True)
        pj = self.proj_dist(obj_id, cam_K, gt_cam_R_m2c, gt_cam_t_m2c, pred_cam_R_m2c, pred_cam_t_m2c)
        pv = self.proj_dist_visible(cam_K, sample.get(sf.gt_coord_3d_roi), sample.get(sf.gt_mask_vis_roi),
                                    gt_cam_R_m2c, gt_cam_t_m2c, pred_cam_R_m2c, pred_cam_t_m2c)
        return re, te, ad, pj, pv

    def add_score(self, obj_id: torch.Tensor, gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor,
                  pred_cam_R_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor, div_diameter: bool = True) \
            -> torch.Tensor:
        """
        ADD
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :param div_diameter: bool
        :return: [N]
        """
        N = len(obj_id)
        add = torch.empty(N, dtype=pred_cam_R_m2c.dtype, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = self.objects_eval[int(obj_id[i])]
            add[i] = obj.average_distance(pred_cam_R_m2c[i], gt_cam_R_m2c[i], pred_cam_t_m2c[i], gt_cam_t_m2c[i])
            if div_diameter:
                add[i] /= obj.diameter
        return add

    def proj_dist(self, obj_id: torch.Tensor, cam_K: torch.Tensor, gt_cam_R_m2c: torch.Tensor,
                  gt_cam_t_m2c: torch.Tensor, pred_cam_R_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor) \
            -> torch.Tensor:
        """
        PROJ
        :param objects_eval: dict[int, ObjMesh]
        :param pred_cam_R_m2c: [N, 3, 3]
        :param pred_cam_t_m2c: [N, 3]
        :return: [N]
        """
        N = len(obj_id)
        dist = torch.empty(N, dtype=pred_cam_R_m2c.dtype, device=pred_cam_R_m2c.device)
        for i in range(N):
            obj = self.objects_eval[int(obj_id[i])]
            dist[i] = obj.average_projected_distance(cam_K[i], pred_cam_R_m2c[i], gt_cam_R_m2c[i],
                                                     pred_cam_t_m2c[i], gt_cam_t_m2c[i])
        return dist

    @staticmethod
    def relative_angle(gt_cam_R_m2c: torch.Tensor, pred_cam_R_m2c: torch.Tensor, degree: bool = False) \
            -> torch.Tensor:
        """
        RE
        :param pred_cam_R_m2c: [..., 3, 3]
        :param degree: bool, use 360 degrees
        :return: [...]
        """
        R_diff = pred_cam_R_m2c @ gt_cam_R_m2c.transpose(-2, -1)  # [..., 3, 3]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]  # [...]
        angle = ((trace.clamp(-1., 3.) - 1.) * .5).acos()  # [...]
        if degree:
            angle *= 180. / torch.pi  # 360 degrees
        return angle  # [...]

    @staticmethod
    def relative_dist(gt_cam_t_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor, cm: bool = False) -> torch.Tensor:
        """
        TE
        :param pred_cam_t_m2c: [.., 3]
        :param cm: bool, use cm instead of meter
        :return:
        """
        dist = torch.linalg.vector_norm(pred_cam_t_m2c - gt_cam_t_m2c, ord=2, dim=-1)  # [...]
        if cm:
            dist *= 100.  # cm
        return dist  # [...]

    @staticmethod
    def proj_dist_visible(cam_K: torch.Tensor, gt_coord_3d_roi: torch.Tensor, gt_mask_vis_roi: torch.Tensor,
                          gt_cam_R_m2c: torch.Tensor, gt_cam_t_m2c: torch.Tensor,
                          pred_cam_R_m2c: torch.Tensor, pred_cam_t_m2c: torch.Tensor) -> torch.Tensor:
        N = len(pred_cam_R_m2c)
        dist = torch.empty(N, dtype=pred_cam_R_m2c.dtype, device=pred_cam_R_m2c.device)
        for i in range(N):
            verts = gt_coord_3d_roi[i].permute(1, 2, 0)[gt_mask_vis_roi[i, 0]]
            dist[i] = ObjMesh.compute_average_projected_distance(verts, cam_K[i], gt_cam_R_m2c[i], pred_cam_R_m2c[i],
                                                                 gt_cam_t_m2c[i], pred_cam_t_m2c[i], p=2)
        return dist
