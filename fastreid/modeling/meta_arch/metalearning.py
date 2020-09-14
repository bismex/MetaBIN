# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
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

    def forward(self, batched_inputs, opt = ''):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            outs = dict()
            outs['targets'] = batched_inputs["targets"].long().to(self.device)
            if self.other_dataset:
                assert "others" in batched_inputs, "View ID annotation are missing in training!"
                assert "dir" in batched_inputs['others'], "View ID annotation are missing in training!"
                outs['views'] = batched_inputs['others']['dir'].long().to(self.device)

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

    def losses(self, outs, opt = ''):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """

        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]

        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']

        loss_names = opt['loss']
        loss_dict = {}
        if self._cfg['META']['GRL']['DO_IT']:
            gt_domains = outs['views']
            dom_outputs = outputs['dom_outputs']
            loss_dict['loss_dom'] = self._cfg['META']['GRL']['WEIGHT'] * \
                                    cross_entropy_loss(
                                        dom_outputs,
                                        gt_domains,
                                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                                    ) * self._cfg.MODEL.LOSSES.CE.SCALE

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

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


        # Reg loss
        if not opt == '':
            if opt['original_learning'] and ("Reg_bottleneck" in loss_names):
                model_w = self.heads.bottleneck.fc.weight
                loss_dict['loss_bottleneck_reg'] = self._cfg.META.LOSS.ORI_REG_WEIGHT * \
                                                   self.heads.bottleneck_reg(torch.abs(torch.flatten(model_w)))[0]
                # print(self.heads.bottleneck_reg.weight[0])
            elif "Reg_bottleneck" in loss_names:
                if self.heads.bottleneck_meta_learning:
                    if opt['param_update']:
                        flag_run = False
                        for name, param in opt['new_param'].items():
                            if 'bottleneck_meta' in name:
                                model_w = param
                                flag_run = True
                                break
                        if not flag_run:
                            model_w = self.heads.bottleneck_meta['view{}'.format(opt['view_idx'])].fc.weight
                    else:
                        model_w = self.heads.bottleneck_meta['view{}'.format(opt['view_idx'])].fc.weight

                    loss_dict['loss_bottleneck_reg'] = self._cfg.META.LOSS.META_REG_WEIGHT * \
                                                       self.heads.bottleneck_reg(torch.abs(torch.flatten(model_w)))[0]

            if opt['original_learning'] and ("Reg_classifier" in loss_names):
                model_w = self.heads.classifier.weight
                loss_dict['loss_classifier_reg'] = self._cfg.META.LOSS.ORI_REG_WEIGHT * \
                                                   self.heads.classifier_reg(torch.abs(torch.flatten(model_w)))[0]
            elif "Reg_classifier" in loss_names:
                if self.heads.classifier_meta_learning:
                    if opt['param_update']:
                        flag_run = False
                        for name, param in opt['new_param'].items():
                            if 'classifier_meta' in name:
                                model_w = param
                                flag_run = True
                                break
                        if not flag_run:
                            model_w = self.heads.classifier_meta['view{}'.format(opt['view_idx'])].weight
                    else:
                        model_w = self.heads.classifier_meta['view{}'.format(opt['view_idx'])].weight

                    loss_dict['loss_classifier_reg'] = self._cfg.META.LOSS.META_REG_WEIGHT * \
                                                       self.heads.classifier_reg(torch.abs(torch.flatten(model_w)))[0]

        return loss_dict
