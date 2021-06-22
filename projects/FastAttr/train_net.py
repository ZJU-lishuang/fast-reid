# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.build import _root, build_reid_train_loader, build_reid_test_loader
from fastreid.data.transforms import build_transforms
from fastreid.utils import comm
from collections import OrderedDict
from fastreid.evaluation import (inference_on_dataset, print_csv_format)

from fastattr import *


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
            setattr(model, "sample_weights1", AttrTrainer.sample_weights[0].to(model.device))
            setattr(model, "sample_weights2", AttrTrainer.sample_weights[1].to(model.device))
            # setattr(model, "sample_weights", AttrTrainer.sample_weights.to(model.device))
        else:
            setattr(model, "sample_weights", None)
        return model

    @classmethod
    def build_train_loader(cls, cfg):

        logger = logging.getLogger("fastreid.attr_dataset")
        train_items = list()
        attr_dict = None
        for d in cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
            if comm.is_main_process():
                dataset.show_train()
            if attr_dict is not None:
                assert attr_dict == dataset.attr_dict, f"attr_dict in {d} does not match with previous ones"
            else:
                attr_dict = dataset.attr_dict
            train_items.extend(dataset.train)

        train_transforms = build_transforms(cfg, is_train=True)
        train_set = AttrDataset(train_items, train_transforms, attr_dict)

        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        AttrTrainer.sample_weights = data_loader.dataset.sample_weights
        return data_loader

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

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            try:
                data_loader, evaluator = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
            results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
            # results[dataset_name] = results_i
            if len(results_i) == 2:
                results["type"] = results_i[0]
                results["country"] = results_i[1]

            if comm.is_main_process():
                assert isinstance(
                    results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                results_i[0]['dataset'] = dataset_name+"_attr1"
                results_i[1]['dataset'] = dataset_name+"_attr2"
                print_csv_format(results_i[0])
                print_csv_format(results_i[1])

        if len(results) == 1:
            results = list(results.values())[0]

        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = AttrTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = AttrTrainer.test(cfg, model)
        return res

    trainer = AttrTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
