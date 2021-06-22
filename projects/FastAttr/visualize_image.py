# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import glob
import os

import cv2

import numpy as np
import torch
import tqdm
from torch.backends import cudnn



sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer
from fastreid.engine import DefaultTrainer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils import comm
from fastreid.data.transforms import build_transforms
from fastreid.data.build import _root
from fastreid.utils.file_io import PathManager
# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

sys.path.append("projects/FastAttr")
from fastattr import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class AttrTrainer(DefaultTrainer):
    sample_weights = None

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = DefaultTrainer.build_model(cfg)
        if cfg.MODEL.LOSSES.BCE.WEIGHT_ENABLED and \
                AttrTrainer.sample_weights is not None:
            setattr(model, "sample_weights", AttrTrainer.sample_weights.to(model.device))
        else:
            setattr(model, "sample_weights", None)
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        attr_dict = dataset.attr_dict
        if comm.is_main_process():
            dataset.show_test()
        test_items = dataset.test

        test_transforms = build_transforms(cfg, is_train=False)
        test_set = AttrDataset(test_items, test_transforms, attr_dict)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, AttrEvaluator(cfg, output_folder)

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    classes_dict1 = {}
    with open("classes1.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict1[idx] = class_name

    classes_dict2 = {}
    with open("classes2.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict2[idx] = class_name

    PathManager.mkdirs(args.output)
    input_path=["../sample/test/*.jpg"]
    save_path=args.output
    input = glob.glob(os.path.expanduser(input_path[0]))
    for path in tqdm.tqdm(input):
        img = cv2.imread(path)
        feat = demo.run_on_image(img)
        feat1=feat[0]
        feat2 = feat[1]
        feat1 = feat1.numpy()
        feat2 = feat2.numpy()
        class_name1=classes_dict1[feat1.argmax()]
        class_name2 = classes_dict2[feat2.argmax()]
        imgname,_=os.path.splitext(os.path.basename(path))
        save_img_name=f"{imgname}_{class_name1}_{class_name2}.jpg"
        cv2.imwrite(os.path.join(save_path,save_img_name),img)




