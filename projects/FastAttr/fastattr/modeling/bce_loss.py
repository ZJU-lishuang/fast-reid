# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F


def ratio2weight(targets, ratio):
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    weights[targets > 1] = 0.0
    return weights


def cross_entropy_sigmoid_loss(pred_class_logits, gt_classes, sample_weight=None):
    loss1 = F.binary_cross_entropy_with_logits(pred_class_logits[0], gt_classes[0], reduction='none')
    loss2 = F.binary_cross_entropy_with_logits(pred_class_logits[1], gt_classes[1], reduction='none')
    batchsize1 = torch.zeros_like(gt_classes[0])
    batchsize2 = torch.zeros_like(gt_classes[1])
    for index,gt_classes1 in enumerate(gt_classes[0]):
        batchsize1[index]=gt_classes1.max()
    for index,gt_classes2 in enumerate(gt_classes[1]):
        batchsize2[index]=gt_classes2.max()
    loss1 = loss1 * batchsize1
    loss2 = loss2 * batchsize2
    if sample_weight is not None:
        targets_mask1 = torch.where(gt_classes[0].detach() > 0.5,
                                   torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
        weight1 = ratio2weight(targets_mask1, sample_weight[0])
        loss1 = loss1 * weight1
        targets_mask2 = torch.where(gt_classes[1].detach() > 0.5,
                                   torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
        weight2 = ratio2weight(targets_mask2, sample_weight[1])
        loss2 = loss2 * weight2

    with torch.no_grad():
        non_zero_cnt1 = max(loss1.nonzero(as_tuple=False).size(0), 1)
        non_zero_cnt2 = max(loss2.nonzero(as_tuple=False).size(0), 1)

    loss1 = loss1.sum() / non_zero_cnt1
    loss2 = loss2.sum() / non_zero_cnt2
    loss = loss1 + loss2
    return loss
