import os.path

import torch
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from matplotlib import pyplot as plt

import augmentations.color_augmentation
import config.const as cc
from dataloader.data_module import LitDataModule
from dataloader.sample import Sample, SampleFields as sf
from models.main_model import MainModel
from utils.config import Config


num_imgs = 30000
split_name = 'train'
root_dir = '/data/105sa'
# ckpt_path = '/data/lightning_logs_archive/104/104_siren_long/version_341/checkpoints/epoch=0142-val_metric=0.3394.ckpt'
ckpt_path = '/data/lightning_logs_archive/105/105_siren_long/version_333/checkpoints/epoch=0149-val_metric=0.3378.ckpt'


def setup_t4p(args=None) -> Config:
    cfg = Config.fromfile('config/top.py')
    if args is not None:
        cfg.merge_from_dict(args)
    return cfg


cfg_t4p = setup_t4p()
datamodule = LitDataModule(cfg_t4p)
dataloader = iter(datamodule.train_dataloader())

if ckpt_path is not None:
    t4p_model = MainModel.load_from_checkpoint(
        ckpt_path, strict=True, cfg=cfg_t4p,
        objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval
    )
else:
    t4p_model = MainModel(cfg_t4p, datamodule.dataset.objects, datamodule.dataset.objects_eval)

t4p_model = t4p_model.to(cc.device)
t4p_model.eval()

coco = Coco()
coco.add_category(CocoCategory(id=0, name='obj'))

split_dir = os.path.join(root_dir, split_name)
os.makedirs(split_dir, exist_ok=True)

with torch.no_grad():
    i = 0
    while i < num_imgs:
        sample: Sample = next(dataloader)
        N, C, H, W = sample.img_roi.shape
        sample.get_gt_coord_3d_roi_normalized()
        t4p_model.forward_texture(sample)
        sample.set(sf.img_roi, augmentations.color_augmentation.match_background_histogram(
            sample.get(sf.img_roi), sample.get(sf.gt_mask_vis_roi), **t4p_model.match_background_histogram_cfg
        ))
        sample.set(sf.img_roi, t4p_model.transform(sample.get(sf.img_roi)))
        for j in range(N):
            file_name = f'{i+j:0>12}.jpg'
            coco_image = CocoImage(file_name=file_name, height=H, width=W)
            x1, y1, x2, y2 = sample.bbox[j]
            x_min, y_min, width, height = x1, y1, x2 - x1, y2 - y1
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[x_min, y_min, width, height],
                    category_id=0,
                    category_name='obj'
                )
            )
            coco.add_image(coco_image)
            plt.imsave(os.path.join(split_dir, file_name),
                       sample.img_roi[j].permute(1, 2, 0).detach().cpu().numpy(),
                       vmin=0., vmax=1.)
        i += N
        print(i)

save_json(data=coco.json, save_path=os.path.join(root_dir, f'{split_name}.json'))
