# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import copy


@META_ARCH_REGISTRY.register()
class Metalearning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        if cfg.META.DATA.NAMES == "":
            self.other_dataset = False
        else:
            self.other_dataset = True

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, opt = None):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images, opt)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            outs = dict()
            outs['targets'] = batched_inputs["targets"].long().to(self.device)
            if self.other_dataset:
                assert "others" in batched_inputs, "View ID annotation are missing in training!"
                assert "domains" in batched_inputs['others'], "View ID annotation are missing in training!"
                outs['domains'] = batched_inputs['others']['domains'].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if outs['targets'].sum() < 0: outs['targets'].zero_()

            outs['outputs'] = self.heads(features, outs['targets'], opt)

            return outs
        else:
            return self.heads(features)

    def preprocess_image(self, batched_inputs, opt = ''):
        """
        Normalize and batch the input images.
        """
        images = batched_inputs["images"].to(self.device)
        # images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
    def losses(self, outs, opt = None):

        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]

        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']

        loss_names = opt['loss']
        loss_dict = {}
        log_accuracy(pred_class_logits, gt_labels) # Log prediction accuracy

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        return loss_dict