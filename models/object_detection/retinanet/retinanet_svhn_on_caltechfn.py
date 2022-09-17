import torch
from tqdm import tqdm
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import csv
import pandas as pd
import h5py

from detectron2 import model_zoo
from detectron2.structures.boxes import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# Register Datasets

from detectron2.data.datasets import register_coco_instances
register_coco_instances("svhn_train", {}, "../../../datasets/svhn/train/train_coco.json", "../../../datasets/svhn/train")
register_coco_instances("caltechfn_test", {}, "../../../datasets/caltechfn/test/train_coco.json", "../../../datasets/caltechfn/test")

# Training
from detectron2.engine import DefaultTrainer

# turn pretraining off, initialized to random weights
cfg = model_zoo.get_config("COCO-Detection/retinanet_R_50_FPN_3x.yaml", trained = False) 
cfg.DATASETS.TRAIN = ("svhn_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = ""
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005  
cfg.SOLVER.MOMENTUM = 0.93
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.MAX_ITER = 30000   # (max_iter * batch_size) / num_images = num_epochs
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("caltechfn_test",cfg,False,output_dir="./output")
test_loader = build_detection_test_loader(cfg, "caltechfn_test")
print(inference_on_dataset(predictor.model, test_loader, evaluator))