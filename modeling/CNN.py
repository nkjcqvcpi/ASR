# encoding: utf-8

from collections import OrderedDict

import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_nl import ResNetNL
from .layer import GeneralizedMeanPoolingP
from .weights_init import weights_init_kaiming, weights_init_classifier


class CNN(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, cfg, logger):
        super(CNN, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.logger = logger

        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            self.logger.info('using resnet50 as a backbone')
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride, block=Bottleneck,
                                 layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3])
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, dropout_p=None,
                              inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, dropout_p=None,
                              inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, layers=[3, 8, 36, 3], groups=1, reduction=16, dropout_p=None,
                              inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, dropout_p=None,
                              inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, dropout_p=None,
                              inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16, dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        else:
            self.logger.error('unsupported backbone! but got {}'.format(model_name))

        self.num_classes = num_classes

        if cfg.MODEL.GENERALIZED_MEAN_POOL:
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)

        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck:  # normalize for angular softmax
            feat = self.bottleneck(global_feat)
        else:
            feat = global_feat

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path, device):
        param_dict = torch.load(trained_path, map_location=torch.device(device))['model']
        if not isinstance(param_dict, OrderedDict):
            param_dict = param_dict.state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        self.logger.info('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        self.logger.info('Loading pretrained model for finetuning from {}'.format(model_path))
