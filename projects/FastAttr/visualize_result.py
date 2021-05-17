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
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
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
    _root="../data"
    dataset = DATASET_REGISTRY.get(args.dataset_name)(root=_root)
    attr_dict = dataset.attr_dict
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.test
    test_transforms = build_transforms(cfg, is_train=False)
    test_set = AttrDataset(test_items, test_transforms, attr_dict)
    test_loader, _ = build_reid_test_loader(cfg, test_set=test_set)

    # test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    # PathManager.mkdirs(args.output)
    # input_path=["/home/lishuang/Disk/gitlab/traincode/car_attr/data/test/*.jpg"]
    # input = glob.glob(os.path.expanduser(input_path[0]))
    # for path in tqdm.tqdm(input):
    #     img = cv2.imread(path)
    #     feat = demo.run_on_image(img)
    #     feat = feat.numpy()
    #     np.save(os.path.join(args.output, os.path.basename(path) + '.npy'), feat)

    logger.info("Start extracting image features")
    feats1 = []
    feats2 = []
    pids1 = []
    pids2 = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats1.append(feat[0])
        pids1.extend(pid[0])
        feats2.append(feat[1])
        pids2.extend(pid[1])
        camids.extend(camid)

    feats1=torch.cat(feats1, dim=0)
    feats2 = torch.cat(feats2, dim=0)

    attr_list=test_loader.dataset.attr_dict
    thr=0.2
    for (feat1, pid1,feat2, pid2, camid) in zip(feats1,pids1,feats2,pids2,camids):
        print("img=",camid)
        # imgattr1=feat1.gt(thr)
        imgattr1=feat1.gt(feat1[feat1.argmax()] - 0.001)
        # imgattr2 = feat2.gt(thr)
        imgattr2 = feat2.gt(feat2[feat2.argmax()] - 0.001)
        for key in attr_list[0]:
            if imgattr1[key]:
                print("type res=",attr_list[0][key])
            if pid1[key]:
                print("type target=",attr_list[0][key])
        for key in attr_list[1]:
            if imgattr2[key]:
                print("country res=",attr_list[1][key])
            if pid2[key]:
                print("country target=",attr_list[1][key])


