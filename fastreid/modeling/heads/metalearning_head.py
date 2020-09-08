# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import logging
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch.autograd import Function

# logger = logging.getLogger(__name__)

@REID_HEADS_REGISTRY.register()
class MetalearningHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer

        # Metalearning parameters
        reduction_dim = cfg.META.BOTTLENECK.REDUCTION_DIM # 1024
        self.BOTTLENECK = cfg.META.BOTTLENECK.DO_IT # True
        self.GRL = cfg.META.GRL.DO_IT # True
        self.GRL_LOCATION = cfg.META.GRL.LOCATION # 'before' / 'after
        self.bottleneck = []
        self.bottleneck_reg = []
        self.GRL_classifier = []

        if self.GRL_LOCATION == 'before':
            GRL_input_dim = in_feat
        else:
            GRL_input_dim = reduction_dim

        if not self.BOTTLENECK:
            self.GRL_LOCATION = 'after'
            reduction_dim = in_feat
            GRL_input_dim = in_feat
        # self.GRL_LOCATION = 'fc' # 'fc' / 'last_conv' / 'init'

        if self.GRL:
            self.GRL_classifier = GRL_classifier(input_dim = GRL_input_dim, bottleneck_dim = 1024, w_lrelu = 0.1, act = 'lrelu', droprate = 0.5, output_dim = 8, n_layer = 2, bnorm = True)
            self.GRL_classifier.apply(weights_init_kaiming)

        # *** meta_learning with bottleneck fc layer ***
        self.bottleneck_meta_learning = False
        if self.BOTTLENECK:
            self.bottleneck = bottleneck_layer(in_feat, reduction_dim, cfg)
            if 'bottleneck' in cfg.META.MODEL.SPLIT_LAYER:
                self.bottleneck_meta_learning = True
                n_view = cfg.META.DATA.NUM_VIEW
                self.bottleneck_meta = nn.ModuleDict()
                for i in range(n_view):
                    self.bottleneck_meta['view{}'.format(i)] = bottleneck_layer(in_feat, reduction_dim, cfg)

                self.bottleneck_reg = nn.Linear(in_feat * reduction_dim, 1, bias=False)
                self.bottleneck_reg.apply(weights_init_kaiming)
        else:
            if 'bottleneck' in cfg.META.MODEL.SPLIT_LAYER:
                print("error in bottleneck")

        # *** meta_learning with classifier fc layer ***
        self.classifier_meta_learning = False
        if 'classifier'in cfg.META.MODEL.SPLIT_LAYER:
            self.classifier_meta_learning = True
            n_view = cfg.META.DATA.NUM_VIEW
            self.classifier_meta = nn.ModuleDict()
            for i in range(n_view):
                self.classifier_meta['view{}'.format(i)] = nn.Linear(reduction_dim, num_classes, bias=False)
                self.classifier_meta['view{}'.format(i)].apply(weights_init_kaiming)
            self.classifier_reg = nn.Linear(reduction_dim * num_classes, 1, bias=False)
            self.classifier_reg.apply(weights_init_kaiming)


        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)

        if cfg.MODEL.HEADS.NORM_INIT_KAIMING:
            self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':          self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, reduction_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, reduction_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax' and 'circleSoftmax'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None, opt = ''):
        """
        See :class:`ReIDHeads.forward`.
        """

        if opt == '':
            opt = {}
            # logger.info('option is blank')
            opt['ds_flag'] = False
            opt['param_update'] = False

        if self.BOTTLENECK:
            features = self.pool_layer(features)
            if self.bottleneck_meta_learning and opt['ds_flag']:
                global_feat = self.bottleneck_meta['view{}'.format(opt['view_idx'])](features[...,0,0], opt)
            else:
                global_feat = self.bottleneck(features[...,0,0], opt)
            global_feat = global_feat.unsqueeze(-1)
            global_feat = global_feat.unsqueeze(-1)
        else:
            # global_feat = self.pool_layer(features[...,0,0])
            global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]


        # Evaluation
        if not self.training: return bn_feat # this feature is inference feature

        if self.GRL:
            if self.GRL_LOCATION == 'before':
                dom_outputs = self.GRL_classifier.forward(f = features[..., 0, 0], alpha = opt['GRL_constant'])
            else:
                dom_outputs = self.GRL_classifier.forward(f = global_feat[..., 0, 0], alpha = opt['GRL_constant'])

        if self.classifier_meta_learning and opt['ds_flag']:

            # cls_outputs = self.classifier_meta['view{}'.format(opt['view_idx'])](bn_feat)

            if opt['param_update']:
                flag_run = False
                for name, param in opt['new_param'].items():
                    if 'classifier_meta' in name:
                        cls_outputs = torch.nn.functional.linear(input = bn_feat, weight = param, bias = None)
                        pred_class_logits = F.linear(bn_feat, param)
                        flag_run = True
                        break
                if not flag_run:
                    cls_outputs = torch.nn.functional.linear(input = bn_feat, weight = self.classifier_meta['view{}'.format(opt['view_idx'])].weight, bias = None)
                    pred_class_logits = F.linear(bn_feat, self.classifier_meta['view{}'.format(opt['view_idx'])].weight)
            else:
                cls_outputs = torch.nn.functional.linear(input = bn_feat, weight = self.classifier_meta['view{}'.format(opt['view_idx'])].weight, bias = None)
                pred_class_logits = F.linear(bn_feat, self.classifier_meta['view{}'.format(opt['view_idx'])].weight)

        else:
            try:              cls_outputs = self.classifier(bn_feat)
            except TypeError: cls_outputs = self.classifier(bn_feat, targets) # for ArcSoftmax & CircleSoftmax
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0] # this feature is triplet feature
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        if self.GRL:
            return cls_outputs, pred_class_logits, feat, dom_outputs
        else:
            return cls_outputs, pred_class_logits, feat

class bottleneck_layer(nn.Module):

    def __init__(self, in_feat, reduction_dim, cfg):
        super(bottleneck_layer, self).__init__()

        self.bn_flag = cfg.META.BOTTLENECK.NORM
        HEAD_NORM = cfg.MODEL.HEADS.BT_NORM
        NORM_SPLIT = cfg.MODEL.HEADS.NORM_SPLIT

        self.fc = nn.Linear(in_feat, reduction_dim, bias=False)
        self.fc.apply(weights_init_kaiming)

        if self.bn_flag:
            self.bn = get_norm(HEAD_NORM, reduction_dim, NORM_SPLIT)
            if cfg.MODEL.HEADS.NORM_INIT_KAIMING:
                self.bn.apply(weights_init_kaiming)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.lrelu.apply(weights_init_kaiming)

    def forward(self, x, opt):

        if not opt == '':
            if opt['param_update']:
                flag_run = False
                for name, param in opt['new_param'].items():
                    if 'bottleneck_meta' in name:
                        x = torch.nn.functional.linear(input = x, weight = param, bias = None)
                        flag_run = True
                        break
                if not flag_run:
                    x = torch.nn.functional.linear(input=x, weight=self.fc.weight, bias=None)
            else:
                x = torch.nn.functional.linear(input = x, weight = self.fc.weight, bias = None)
        else:
            x = torch.nn.functional.linear(input=x, weight=self.fc.weight, bias=None)

        if self.bn_flag:
            x = x.unsqueeze(-1)
            x = x.unsqueeze(-1)
            x = self.bn(x)
            x = x.squeeze()

        x = self.lrelu(x)

        return x

        # Define the ResNet50-based Model
class GRL_classifier(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, output_dim, n_layer, bnorm, droprate, act, w_lrelu):
        super(GRL_classifier, self).__init__()

        # fc(1024)->bn->lrelu->dp->fc(512)->bn->lrelu->dp->fc(8)

        add_block = []
        for i in range(n_layer):
            add_block += [nn.Linear(input_dim, bottleneck_dim, bias=False)]
            if bnorm:
                add_block += [nn.BatchNorm1d(bottleneck_dim)]
            if act == 'relu':
                add_block += [nn.ReLU(inplace=True)]
            elif act == 'lrelu':
                add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
            elif act == 'prelu':
                add_block += [nn.PReLU()]
            elif act == 'selu':
                add_block += [nn.SELU(inplace=True)]
            elif act == 'tanh':
                add_block += [nn.Tanh()]
            elif act == 'none':
                print('.')
            if droprate:
                add_block += [nn.Dropout(p=droprate)]
            input_dim = bottleneck_dim
            bottleneck_dim = bottleneck_dim // 2
        add_block += [nn.Linear(input_dim, output_dim, bias=False)]

        add_block = nn.Sequential(*add_block)
        self.domain_classifier = add_block


    def forward(self, f, alpha):
        reverse_f = ReverseLayerF.apply(f, alpha) # [128, 800]
        domain_output = self.domain_classifier(reverse_f)

        return domain_output

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

