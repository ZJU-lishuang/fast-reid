# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import torch

from fastreid.evaluation.evaluator import DatasetEvaluator
from fastreid.utils import comm

logger = logging.getLogger("fastreid.attr_evaluation")


class AttrEvaluator(DatasetEvaluator):
    def __init__(self, cfg, attr_dict, thres=0.5, output_dir=None):
        self.cfg = cfg
        self.attr_dict = attr_dict
        self.thres = thres
        self._output_dir = output_dir

        self.pred_logits1 = []
        self.gt_labels1 = []
        self.pred_logits2 = []
        self.gt_labels2 = []
        self.pred_logits = []
        self.gt_labels = []

    def reset(self):
        self.pred_logits1 = []
        self.gt_labels1 = []
        self.pred_logits2 = []
        self.gt_labels2 = []
        self.pred_logits = []
        self.gt_labels = []

    def process(self, inputs, outputs):
        self.gt_labels1.extend(inputs["targets1"].cpu())
        self.gt_labels2.extend(inputs["targets2"].cpu())
        self.pred_logits1.extend(outputs[0].cpu())
        self.pred_logits2.extend(outputs[1].cpu())
        self.gt_labels=[self.gt_labels1,self.gt_labels2]
        self.pred_logits = [self.pred_logits1, self.pred_logits2]

    @staticmethod
    def get_attr_metrics(gt_labels, pred_logits, thres):

        eps = 1e-20

        pred_labels = copy.deepcopy(pred_logits)
        pred_labels[pred_logits < thres] = 0
        pred_labels[pred_logits >= thres] = 1

        # Compute label-based metric
        overlaps = pred_labels * gt_labels
        correct_pos = overlaps.sum(axis=0)
        real_pos = gt_labels.sum(axis=0)
        inv_overlaps = (1 - pred_labels) * (1 - gt_labels)
        correct_neg = inv_overlaps.sum(axis=0)
        real_neg = (1 - gt_labels).sum(axis=0)

        # Compute instance-based accuracy
        pred_labels = pred_labels.astype(bool)
        gt_labels = gt_labels.astype(bool)
        intersect = (pred_labels & gt_labels).astype(float)
        union = (pred_labels | gt_labels).astype(float)
        ins_acc = (intersect.sum(axis=1) / (union.sum(axis=1) + eps)).mean()
        ins_prec = (intersect.sum(axis=1) / (pred_labels.astype(float).sum(axis=1) + eps)).mean()
        ins_rec = (intersect.sum(axis=1) / (gt_labels.astype(float).sum(axis=1) + eps)).mean()
        ins_f1 = (2 * ins_prec * ins_rec) / (ins_prec + ins_rec + eps)

        term1 = correct_pos / (real_pos + eps)
        term2 = correct_neg / (real_neg + eps)
        label_mA_verbose = (term1 + term2) * 0.5
        label_mA = label_mA_verbose.mean()

        results = OrderedDict()
        results["Accu"] = ins_acc * 100
        results["Prec"] = ins_prec * 100
        results["Recall"] = ins_rec * 100
        results["F1"] = ins_f1 * 100
        results["mA"] = label_mA * 100
        results["metric"] = label_mA * 100
        return results

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            pred_logits1 = comm.gather(self.pred_logits1)
            pred_logits1 = sum(pred_logits1, [])
            pred_logits2 = comm.gather(self.pred_logits1)
            pred_logits2 = sum(pred_logits2, [])

            gt_labels1 = comm.gather(self.gt_labels1)
            gt_labels1 = sum(gt_labels1, [])
            gt_labels2 = comm.gather(self.gt_labels2)
            gt_labels2 = sum(gt_labels2, [])

            if not comm.is_main_process():
                return {}
        else:
            pred_logits1 = self.pred_logits1
            gt_labels1 = self.gt_labels1
            pred_logits2 = self.pred_logits2
            gt_labels2 = self.gt_labels2

        pred_logits1 = torch.stack(pred_logits1, dim=0).numpy()
        gt_labels1 = torch.stack(gt_labels1, dim=0).numpy()
        pred_logits2 = torch.stack(pred_logits2, dim=0).numpy()
        gt_labels2 = torch.stack(gt_labels2, dim=0).numpy()

        # Pedestrian attribute metrics
        thres = self.cfg.TEST.THRES
        self._results1 = self.get_attr_metrics(gt_labels1, pred_logits1, thres)
        self._results2 = self.get_attr_metrics(gt_labels2, pred_logits2, thres)
        self._results=[self._results1,self._results2]

        return copy.deepcopy(self._results)
