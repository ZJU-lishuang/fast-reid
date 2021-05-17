# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.modeling.heads import EmbeddingHead
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_kaiming


@REID_HEADS_REGISTRY.register()
class AttrHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        num_types = cfg.MODEL.HEADS.NUM_TYPES
        num_type_classes = cfg.MODEL.HEADS.NUM_TYPE_CLASSES
        assert len(num_type_classes) == num_types

        # self.bnneck = nn.BatchNorm1d(num_classes)
        # self.bnneck.apply(weights_init_kaiming)

        feat_dim=self.weight.shape[1]

        weight_types_tmp=[]
        bnneck_types_tmp=[]
        for num_type_class in num_type_classes:
            weight_types_tmp.append(nn.Parameter(torch.normal(0, 0.01, (num_type_class, feat_dim))))
            bnneck = nn.BatchNorm1d(num_type_class)
            bnneck.apply(weights_init_kaiming)
            bnneck_types_tmp.append(bnneck)

        self.bnneck_types=nn.Sequential(*bnneck_types_tmp)
        self.weight_types = nn.ParameterList(weight_types_tmp)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        logits_types=[]
        for type_id in range(len(self.weight_types)):
            logits = F.linear(neck_feat, self.weight_types[type_id])
            logits = self.bnneck_types[type_id](logits)
            logits_types.append(logits)

        # Evaluation
        if not self.training:
            cls_outptus=[]
            for logits_id in range(len(logits_types)):
                cls_outptus.append(torch.sigmoid(logits_types[logits_id]))
            return cls_outptus

        return {
            "cls_outputs": logits_types,
        }




        # logits = F.linear(neck_feat, self.weight)
        # logits = self.bnneck(logits)
        #
        # # Evaluation
        # if not self.training:
        #     cls_outptus = torch.sigmoid(logits)
        #     return cls_outptus
        #
        # return {
        #     "cls_outputs": logits,
        # }
