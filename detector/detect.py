# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF

import os
import pickle
import random

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

import utils.io


def random_visualize_dataset():
    """
    To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
    """
    dataset_dicts = DatasetCatalog.get("my_train")
    my_metadata = MetadataCatalog.get('my_train')
    for d in dataset_dicts[:1]:
        img = utils.io.imread(d["file_name"], opencv_bgr=False)
        visualizer = Visualizer(img, metadata=my_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image())
        plt.show()


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = os.path.join('outputs', 'detectron2_logs', f'version_{2}', 'model_final.pth')
    cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 60000 // 8  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = os.path.join('outputs', 'detectron2_logs', f'version_{2}')
    return cfg


def do_train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def eval_synt(cfg, predictor):
    dataset_dicts = DatasetCatalog.get("my_train")
    my_metadata = MetadataCatalog.get('my_train')
    for d in random.sample(dataset_dicts, 3):
        im = utils.io.imread(d["file_name"], opencv_bgr=True)
        outputs = predictor(im)
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1], metadata=my_metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()

    """
    We can also evaluate its performance using AP metric implemented in COCO API.
    """
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("my_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


def eval_real(predictor):
    img_path_list = utils.io.list_img_from_dir('/data/real_exp/i12P_26mm/000105/siren', ext='heic')
    outputs_list = []
    for img_path in tqdm(img_path_list):
        im = utils.io.imread(img_path, opencv_bgr=False)
        outputs = predictor(im[:, :, ::-1])
        outputs_list.append(outputs)

        v = Visualizer(im, metadata=None, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imsave(
            os.path.join('/home/user/Desktop/tmp', f'{img_path.split("/")[-1].split(".")[0]}.jpg'),
            out.get_image(),
            vmin=0., vmax=1.,
        )
        plt.imshow(out.get_image())

        plt.show()

    with open('outputs/detections.pkl', 'wb') as f:
        pickle.dump(outputs_list, f)


def main():
    setup_logger()

    register_coco_instances("my_train", {}, "/data/105sa/train.json", "/data/105sa/train")
    register_coco_instances("my_val", {}, "/data/105sa/val.json", "/data/105sa/val")

    # random_visualize_dataset()

    cfg = make_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    do_train(cfg)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_{6999:>07}.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    eval_synt(cfg, predictor)
    # eval_real(predictor)


if __name__ == '__main__':
    main()
