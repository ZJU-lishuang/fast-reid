# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .bce_loss import cross_entropy_sigmoid_loss


@META_ARCH_REGISTRY.register()
class AttrBaseline(Baseline):

    @classmethod
    def from_config(cls, cfg):
        base_res = Baseline.from_config(cfg)
        base_res["loss_kwargs"].update({
            'bce': {
                'scale': cfg.MODEL.LOSSES.BCE.SCALE
            }
        })
        return base_res

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets1" in batched_inputs, "Person ID annotation are missing in training!"
            assert "targets2" in batched_inputs, "Person ID annotation are missing in training!"
            targets1 = batched_inputs["targets1"]
            targets2 = batched_inputs["targets2"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets1.sum() < 0: targets1.zero_()
            if targets2.sum() < 0: targets2.zero_()
            targets=[targets1,targets2]
            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        cls_outputs = outputs["cls_outputs"]

        loss_dict = {}
        loss_names = self.loss_kwargs["loss_names"]

        if "BinaryCrossEntropyLoss" in loss_names:
            bce_kwargs = self.loss_kwargs.get('bce')
            loss_dict["loss_bce"] = cross_entropy_sigmoid_loss(
                cls_outputs,
                gt_labels,
                [self.sample_weights1,self.sample_weights2],
            ) * bce_kwargs.get('scale')

        return loss_dict
