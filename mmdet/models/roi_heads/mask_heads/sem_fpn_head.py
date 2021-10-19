# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
import numpy as np

@HEADS.register_module()
class SemFPNhead(BaseModule):

    def __init__(self,
                 in_channels=[256, 256, 256, 256],
                 feature_strides=[4, 8, 16, 32],    
                 channels=128,
                 num_classes=2,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 ignore_label=None,
                 loss_weight=None,
                 align_corners = False,
                 loss_seg=dict(type='CrossEntropyLoss',ignore_index=255,loss_weight=0.2),
                 init_cfg=dict(
                     #type='Kaiming', override=dict(name='conv_logits')
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
  
        super(SemFPNhead, self).__init__(init_cfg)
        
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg= act_cfg
        self.in_channels = in_channels
        self.align_corners = align_corners
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
            
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
        if ignore_label:
            loss_seg['ignore_index'] = ignore_label
        if loss_weight:
            loss_seg['loss_weight'] = loss_weight
        if ignore_label or loss_weight:
            warnings.warn('``ignore_label`` and ``loss_weight`` would be '
                          'deprecated soon. Please set ``ingore_index`` and '
                          '``loss_weight`` in ``loss_seg`` instead.')
        self.criterion = build_loss(loss_seg)
        
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output        
    
    @auto_fp16()
    def forward(self, feats):
        
        x = feats[:-1]
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                    self.scale_heads[i](x[i]), size=output.shape[2:], mode='bilinear', align_corners=self.align_corners)
            

        mask_pred = self.cls_seg(output)   
        
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        return loss_semantic_seg
