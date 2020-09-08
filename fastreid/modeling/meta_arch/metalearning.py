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
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg) # conv layer
        # self.backbone_ori = copy.deepcopy(self.backbone)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)
        # self.heads_ori = copy.deepcopy(self.heads)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, opt = ''):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)


            assert "others" in batched_inputs, "View ID annotation are missing in training!"
            assert "dir" in batched_inputs['others'], "View ID annotation are missing in training!"
            views = batched_inputs['others']['dir'].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets, opt)

            return outputs, targets, views
        else:
            outputs = self.heads(features)

            return outputs

    def preprocess_image(self, batched_inputs, opt = ''):
        """
        Normalize and batch the input images.
        """
        images = batched_inputs["images"].to(self.device)
        # images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, targets, opt = ''):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """

        loss_dict = {}
        if self._cfg['META']['GRL']['DO_IT']:
            gt_labels = targets['IDs']
            gt_domains = targets['views']
            cls_outputs, pred_class_logits, pred_features, dom_outputs = outputs
            loss_dict['loss_dom'] = self._cfg['META']['GRL']['WEIGHT'] * CrossEntropyLoss(self._cfg)(dom_outputs, gt_domains) # CE same (0.1 eps, weight 1)
        else:
            cls_outputs, pred_class_logits, pred_features = outputs
            gt_labels = targets

        loss_names = opt['loss']

        # Log prediction accuracy
        CrossEntropyLoss.log_accuracy(pred_class_logits.detach(), gt_labels)

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = CrossEntropyLoss(self._cfg)(cls_outputs, gt_labels)

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = TripletLoss(self._cfg)(pred_features, gt_labels)

        if not opt == '':
            if opt['original_learning'] and ("Reg_bottleneck" in loss_names):
                model_w = self.heads.bottleneck.fc.weight
                loss_dict['loss_bottleneck_reg'] = self._cfg.META.LOSS.ORI_REG_WEIGHT * self.heads.bottleneck_reg(torch.abs(torch.flatten(model_w)))[0]
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

                    loss_dict['loss_bottleneck_reg'] = self._cfg.META.LOSS.META_REG_WEIGHT * self.heads.bottleneck_reg(torch.abs(torch.flatten(model_w)))[0]
                    # print('reg: {}'.format(loss_dict['loss_bottleneck_reg']))

            if opt['original_learning'] and ("Reg_classifier" in loss_names):
                model_w = self.heads.classifier.weight
                loss_dict['loss_classifier_reg'] = self._cfg.META.LOSS.ORI_REG_WEIGHT * self.heads.classifier_reg(torch.abs(torch.flatten(model_w)))[0]
                # print(self.heads.bottleneck_reg.weight[0])
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

                    loss_dict['loss_classifier_reg'] = self._cfg.META.LOSS.META_REG_WEIGHT * self.heads.classifier_reg(torch.abs(torch.flatten(model_w)))[0]
                    # print('reg: {}'.format(loss_dict['loss_bottleneck_reg']))

        return loss_dict
