import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import augmentations.color_augmentation
import config.const as cc
from dataloader.data_module import LitDataModule
from dataloader.sample import Sample, SampleFields as sf
from models.main_model import MainModel
from utils.config import Config
from utils.io import imread

from detector.plain_train_net import do_train


setup_logger()


def get_balloon_dicts():
    dataset_dicts = []
    record = {}

    record["file_name"] = '/data/dummy512.jpg'
    record["image_id"] = 123
    record["height"] = 512
    record["width"] = 512

    objs = []
    obj = {
        "bbox": [0., 0., 512., 512.],
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": 0,
    }
    objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return dataset_dicts * 100


DatasetCatalog.register("balloon", get_balloon_dicts)
MetadataCatalog.get("balloon").set(thing_classes=["balloon"])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 1  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


class Model(nn.Module):
    def __init__(self, detectron_model: nn.Module, t4p_model: MainModel, dataloader: DataLoader):
        super().__init__()
        self.detectron_model: nn.Module = detectron_model
        self.t4p_model: MainModel = t4p_model
        self.dataloader = iter(dataloader)

    def forward(self, data):
        self.t4p_model.eval()
        sample: Sample = next(self.dataloader)
        N, C, H, W = sample.img_roi.shape
        with torch.no_grad():
            sample.get_gt_coord_3d_roi_normalized()
            self.t4p_model.forward_texture(sample)
            for i in range(len(data)):
                sample.set(sf.img_roi, augmentations.color_augmentation.match_background_histogram(
                    sample.get(sf.img_roi), sample.get(sf.gt_mask_vis_roi), **self.t4p_model.match_background_histogram_cfg
                ))
                sample.set(sf.img_roi, self.t4p_model.transform(sample.get(sf.img_roi)))

                d = data[i]
                d['image'] = (sample.img_roi[i] * 255.).round().to(dtype=torch.uint8)
                d['height'] = H
                d['width'] = W
                d['instances']._image_size = (H, W)
                d['instances']._fields['gt_boxes'].tensor = sample.bbox[i][None]
                d['instances']._fields['gt_classes'] = torch.zeros(1, dtype=torch.long)
            #     print(d['instances'])
            #     x1, y1, x2, y2 = d['instances']._fields['gt_boxes'].tensor[0].int()
            #     sample.img_roi[i, :, y1:y2, x1:x2] = 0
            # sample.visualize()
            # a = 0
        return self.detectron_model(data)


def setup_t4p(args=None) -> Config:
    cfg = Config.fromfile('config/top.py')
    if args is not None:
        cfg.merge_from_dict(args)
    return cfg


cfg_t4p = setup_t4p()
cfg.SOLVER.IMS_PER_BATCH = cfg_t4p.dataset.num_obj * cfg_t4p.dataloader.batch_size
datamodule = LitDataModule(cfg_t4p)
dataloader = datamodule.train_dataloader()

ckpt_path = '/data/lightning_logs_archive/104/104_siren_long/version_341/checkpoints/epoch=0142-val_metric=0.3394.ckpt'
if ckpt_path is not None:
    t4p_model = MainModel.load_from_checkpoint(
        ckpt_path, strict=True, cfg=cfg_t4p,
        objects=datamodule.dataset.objects, objects_eval=datamodule.dataset.objects_eval
    )
else:
    t4p_model = MainModel(cfg_t4p, datamodule.dataset.objects, datamodule.dataset.objects_eval)
model = Model(build_model(cfg), t4p_model, dataloader).to(cc.device)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# do_train(cfg, model, resume=False)

predictor = DefaultPredictor(cfg)

state_dict = {}
for k, v in torch.load('output/model_final.pth')['model'].items():
    if k.startswith('detectron_model.'):
        state_dict[k[len('detectron_model.'):]] = v

predictor.model.load_state_dict(state_dict, strict=True)

im = imread('/data/real_exp/i12P_26mm/000104/siren/IMG_9612.HEIC', opencv_bgr=False)

sample: Sample = next(iter(dataloader))
sample.get_gt_coord_3d_roi_normalized()
t4p_model.forward_texture(sample)
im = (sample.img_roi[0].permute(1, 2, 0) * 255.).round().to(torch.uint8).detach().cpu().numpy()

outputs = predictor(im)
v = Visualizer(im, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image())
plt.show()
