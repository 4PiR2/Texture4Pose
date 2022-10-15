import copy
from collections import namedtuple
from typing import Callable, Optional, Union

import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import utils.image_2d
import utils.transform_3d


def not_none(l):
    flag = True
    if l is None:
        flag = False
    elif isinstance(l, list):
        if len(l) == 0:
            flag = False
        else:
            flag = sum([o is None for o in l]) == 0
    return flag


_fields_str = [
    'N', 'gt_cam_t_m2c_site', 'gt_coord_3d_roi_normalized',
    'o_scene', 'o_item', 'code_info',
    'obj_id', 'obj_size', 'obj_diameter', 'cam_K', 'cam_K_orig', 'gt_cam_R_m2c', 'gt_cam_t_m2c', 'bbox',
    'gt_bbox_vis', 'gt_vis_ratio', 'gt_cam_R_m2c_aug',
    'img', 'img_roi', 'coord_2d', 'coord_2d_roi', 'gt_coord_3d', 'gt_coord_3d_roi', 'gt_normal', 'gt_normal_roi',
    'gt_mask_vis', 'gt_mask_vis_roi', 'gt_mask_obj', 'gt_mask_obj_roi', 'gt_light_texel', 'gt_texel_roi',
    'gt_light_specular', 'gt_light_specular_roi', 'gt_zbuf', 'gt_texel', 'gt_light_texel_roi', 'gt_bg', 'gt_bg_roi',
    'pred_cam_R_m2c', 'pred_cam_t_m2c', 'pred_cam_t_m2c_site', 'pred_coord_3d_roi', 'pred_coord_3d_roi_normalized',
    'pred_mask_vis_roi', 'pred_weight_2d', 'pnp_cam_R_m2c', 'pnp_cam_t_m2c', 'pnp_inlier_roi',
]
sf = SampleFields = namedtuple('SampleFields', _fields_str)(*_fields_str)


class Sample:
    def __init__(self, *samples, **kwargs):
        if samples:
            Sample.collate(samples, self)
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @classmethod
    def collate(cls, batch: Union[list, tuple], out=None):
        assert len(batch)
        if out is None:
            out = cls()
        for key, value in batch[0].__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(out, key, torch.cat([getattr(b, key) for b in batch], dim=0))
            elif isinstance(value, list):
                setattr(out, key, [v for b in batch for v in getattr(b, key)])
            else:
                assert len(set(getattr(b, key) for b in batch)) == 1
                setattr(out, key, value)
        return out

    def __iter__(self):
        keys = self.__dict__.keys()
        values = self.__dict__.values()
        for value in zip(*values):
            out = Sample()
            for key, v in zip(keys, value):
                if isinstance(v, torch.Tensor):
                    setattr(out, key, v[None])
                elif isinstance(v, list):
                    setattr(out, key, [v])
                else:
                    setattr(out, key, v)
            yield out

    def __len__(self) -> int:
        l = []
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                l.append(len(value))
        return min(l) if l else 0

    @property
    def N(self):
        return len(self)

    def __repr__(self) -> str:
        return f'{len(self)}'

    # def __getattr__(self, attribute_name):
    #     return None

    def clone(self, detach: bool = False):
        out = Sample()
        for key in [key for key in dir(self) if not key.startswith('__') and not callable(getattr(self, key))]:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                if detach:
                    setattr(out, key, value.detach())
                else:
                    setattr(out, key, value.clone())
            else:
                value_copy = copy.deepcopy(value)
                try:
                    setattr(out, key, value_copy)
                except AttributeError:  # is a property
                    pass
        return out

    def get(self, keys: Union[list[str], str], assertion: bool = True):
        if isinstance(keys, str):
            return self.get([keys], assertion)[0]
        values = []
        for key in keys:
            value = None
            try:
                value = getattr(self, key)
            except AttributeError:
                pass
            except AssertionError as e:
                if assertion:
                    raise e
            values.append(value)
        if assertion:
            assert not_none(values)
        return values

    def set(self, keys: Union[list[str], str], values: Union[list, object]):
        if isinstance(keys, str):
            return self.set([keys], [values])
        for key, value in zip(keys, values):
            setattr(self, key, value)

    def compute_pnp(self, sanity_check_mode: bool = False, store: bool = True, ransac: bool = True,
                erode_min: float = 0., erode_max: float = torch.inf) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt_coord_3d_roi, coord_2d_roi, gt_mask_vis_roi = self.get(
            [sf.gt_coord_3d_roi, sf.coord_2d_roi, sf.gt_mask_vis_roi])
        if sanity_check_mode:
            pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi = utils.transform_3d.solve_pnp(gt_coord_3d_roi,
                coord_2d_roi, gt_mask_vis_roi, ransac)
        else:
            pred_coord_3d_roi, pred_mask_vis_roi = self.get([sf.pred_coord_3d_roi, sf.pred_mask_vis_roi])
            pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi = utils.transform_3d.solve_pnp(pred_coord_3d_roi,
                coord_2d_roi, utils.image_2d.erode_mask(pred_mask_vis_roi, erode_min, erode_max), ransac)
        if store:
            self.set([sf.pnp_cam_R_m2c, sf.pnp_cam_t_m2c, sf.pnp_inlier_roi],
                     [pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi])
        return pnp_cam_R_m2c, pnp_cam_t_m2c, pnp_inliers_roi

    def get_roi_zoom_in_ratio(self) -> torch.Tensor:
        feature_roi, bbox = self.get([sf.img_roi, sf.bbox])
        feature_roi_size = max(feature_roi.shape[-2:])
        crop_size = utils.image_2d.get_dzi_crop_size(bbox)
        roi_zoom_in_ratio = feature_roi_size / crop_size  # [N]
        self.set('roi_zoom_in_ratio', roi_zoom_in_ratio)
        return roi_zoom_in_ratio

    def _get_cam_t_m2c_site(self, cam_t_m2c: torch.Tensor) -> torch.Tensor:
        bbox, cam_K, obj_diameter = self.get([sf.bbox, sf.cam_K, sf.obj_diameter])
        return utils.transform_3d.t_to_t_site(cam_t_m2c, bbox, cam_K, obj_diameter)  # [N, 3]

    def _get_cam_t_m2c(self, cam_t_m2c_site: torch.Tensor) -> torch.Tensor:
        bbox, cam_K, obj_diameter = self.get([sf.bbox, sf.cam_K, sf.obj_diameter])
        return utils.transform_3d.t_site_to_t(cam_t_m2c_site, bbox, cam_K, obj_diameter)
        # [N, 3]

    def get_gt_cam_t_m2c_site(self) -> torch.Tensor:
        gt_cam_t_m2c_site = self._get_cam_t_m2c_site(self.get(sf.gt_cam_t_m2c))  # [N, 3]
        self.set(sf.gt_cam_t_m2c_site, gt_cam_t_m2c_site)
        return gt_cam_t_m2c_site  # [N, 3]

    def get_pred_cam_t_m2c(self) -> torch.Tensor:
        pred_cam_t_m2c = self._get_cam_t_m2c(self.get(sf.pred_cam_t_m2c_site))  # [N, 3]
        self.set(sf.pred_cam_t_m2c, pred_cam_t_m2c)
        return pred_cam_t_m2c  # [N, 3]

    def _get_coord_3d_roi_normalized(self, coord_3d_roi: torch.Tensor) -> torch.Tensor:
        obj_size = self.get(sf.obj_size)
        return utils.transform_3d.normalize_coord_3d(coord_3d_roi, obj_size)  # [N, 3(XYZ), H, W]

    def _get_coord_3d_roi(self, coord_3d_roi_normalized: torch.Tensor) -> torch.Tensor:
        obj_size = self.get(sf.obj_size)
        return utils.transform_3d.denormalize_coord_3d(coord_3d_roi_normalized, obj_size)  # [N, 3(XYZ), H, W]

    def get_gt_coord_3d_roi_normalized(self) -> torch.Tensor:
        gt_coord_3d_roi_normalized = self._get_coord_3d_roi_normalized(self.get(sf.gt_coord_3d_roi))
        # [N, 3(XYZ), H, W]
        self.set(sf.gt_coord_3d_roi_normalized, gt_coord_3d_roi_normalized)
        return gt_coord_3d_roi_normalized  # [N, 3(XYZ), H, W]

    def get_pred_coord_3d_roi(self) -> torch.Tensor:
        pred_coord_3d_roi = self._get_coord_3d_roi(self.get(sf.pred_coord_3d_roi_normalized))  # [N, 3(XYZ), H, W]
        self.set(sf.pred_coord_3d_roi, pred_coord_3d_roi)
        return pred_coord_3d_roi

    def visualize(self, return_figs: bool = False, max_samples: int = None) -> Optional[list[Figure]]:
        figs = []

        def draw_fields(ax: Axes, keys: Union[list[str], str], i: int, title: str = '', fn: Callable = lambda *x: x,
                        fn_ax: Callable = utils.image_2d.draw_ax):
            ax.set_title(title)
            values = self.get(keys, False)
            if not_none(values):
                tensors = fn(*[value[i] for value in values])
                if isinstance(tensors, torch.Tensor):
                    fn_ax(ax, tensors)
                else:
                    fn_ax(ax, *tensors)
            else:
                ax.set_aspect('equal')

        sf = SampleFields
        obj_id = self.get([sf.obj_id])[0]
        for i in range(min(max_samples, len(obj_id)) if max_samples is not None else len(obj_id)):
            fig, axs = plt.subplots(3, 5, figsize=(15, 9))

            draw_fields(axs[0, 0], [sf.img_roi], i, 'rendered image')
            draw_fields(axs[1, 0], [sf.gt_texel_roi], i, 'texture')
            draw_fields(axs[2, 0], [sf.gt_normal_roi], i, 'gt normal', lambda x: x * .5 + .5)

            draw_fields(axs[0, 1], [sf.gt_coord_3d_roi_normalized], i, 'gt 3D coord (relative)')
            draw_fields(axs[1, 1], ['pred_coord_3d_roi_normalized'], i, 'pred 3D coord (relative)',
                        lambda x: x.clamp(0., 1.))

            draw_fields(axs[0, 2], [sf.gt_mask_vis_roi], i, 'gt mask vis')
            draw_fields(axs[1, 2], [sf.pred_mask_vis_roi], i, 'pred mask vis', lambda x: x.clamp(0., 1.))

            draw_fields(axs[0, 3], [sf.coord_2d_roi], i, '2D coord (relative)', utils.image_2d.normalize_channel)
            draw_fields(axs[1, 3], [sf.pred_weight_2d], i, 'pred weight 2D (relative)', lambda x: x.clamp(0., 1.))

            draw_fields(axs[0, 4], [sf.cam_K, sf.gt_cam_R_m2c, sf.gt_cam_t_m2c, sf.obj_size, sf.bbox], i, 'gt pose',
                        fn_ax=utils.transform_3d.show_pose)
            draw_fields(axs[1, 4], [sf.cam_K, sf.pred_cam_R_m2c, sf.pred_cam_t_m2c, sf.obj_size, sf.bbox], i,
                        'pred pose', fn_ax=utils.transform_3d.show_pose)

            draw_fields(
                axs[2, 1], [sf.gt_coord_3d_roi, sf.gt_mask_vis_roi, sf.pred_coord_3d_roi], i, 'diff 3D coord (L2)',
                lambda gc, gm, pc: torch.linalg.vector_norm((pc - gc) * gm, dim=-3),
                lambda ax, diff: utils.image_2d.draw_ax_diff(ax, diff, thresh_min=0., thresh_max=1e-2, log_mode=False)
            )
            draw_fields(
                axs[2, 2], [sf.gt_mask_vis_roi, sf.pred_mask_vis_roi], i, 'diff mask (abs)',
                lambda gm, pm: (pm.clamp(0., 1.) - gm.to(dtype=pm.dtype)).abs(),
                lambda ax, diff: utils.image_2d.draw_ax_diff(ax, diff, thresh_min=1e-4, thresh_max=1., log_mode=True)
            )

            draw_fields(axs[2, 3], [sf.pnp_inlier_roi], i, 'pnp inlier', lambda x: x.expand(3, -1, -1))
            draw_fields(axs[2, 4], [sf.cam_K, sf.pnp_cam_R_m2c, sf.pnp_cam_t_m2c, sf.obj_size, sf.bbox], i,
                        'pnp pose', fn_ax=utils.transform_3d.show_pose)

            fig.tight_layout()
            if return_figs:
                figs.append(fig)
            else:
                plt.show()

        return figs if return_figs else None
