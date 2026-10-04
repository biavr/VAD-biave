"""Lightweight 2.5D BEV occupancy head, adapted from FlashOcc's BEVOCCHead2D.

FlashOcc's head lifts a 2D BEV feature map into a full 3D voxel grid (one
semantic class per (x, y, z) voxel) via a channel-to-height expansion trick,
to avoid 3D convolutions. VAD's occupancy ground truth
(tools/data_converter/occupancy_gt_generator.py) is a 2.5D BEV grid instead
-- one semantic class plus one height value per (x, y) cell, not a full
voxel volume -- so this head is simplified accordingly: two plain 1x1 conv
heads (classification + height regression) directly on the BEV feature map.
The channel-to-height expansion, the Dz dimension, and FlashOcc's
camera-visibility masking are all dropped as unnecessary for this target.

This head is purely an auxiliary branch: it reuses the BEV feature map VAD's
own VADHead already computes (`outs['bev_embed']`) and contributes an
additional loss term, so it does not touch the image backbone, BEV encoder,
or any of VAD's existing detection/map/motion decoders.
"""
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

# Sentinel for BEV cells with no lidar points (see occupancy_gt_generator.py).
# Cells with this class are excluded from both the classification and height
# losses below.
IGNORE_CLASS = 255


@MODELS.register_module()
class VADOccHead(BaseModule):
    """Predicts a semantic class and a height value per BEV cell.

    Args:
        in_channels (int): Channels of the input BEV feature map (must match
            the detection/map head's `embed_dims`).
        feat_channels (int): Hidden channels of the shared conv trunk.
        num_classes (int): Number of lidarseg classes (32 for the official
            nuScenes-lidarseg table; IGNORE_CLASS is handled separately, not
            counted here).
        loss_cls (dict): Config for the per-cell classification loss.
        loss_height (dict): Config for the per-cell height regression loss.
    """

    def __init__(self,
                in_channels,
                feat_channels=256,
                num_classes=32,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    ignore_index=IGNORE_CLASS,
                    avg_non_ignore=True,
                    loss_weight=1.0),
                loss_height=dict(type='mmdet.L1Loss', loss_weight=1.0),
                init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.shared_conv = ConvModule(
            in_channels,
            feat_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'))
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        self.height_pred = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_height = MODELS.build(loss_height)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat (Tensor): (B, C, H, W) BEV feature map.

        Returns:
            cls_logits (Tensor): (B, num_classes, H, W).
            height_pred (Tensor): (B, H, W).
        """
        feat = self.shared_conv(bev_feat)
        cls_logits = self.cls_pred(feat)
        height_pred = self.height_pred(feat).squeeze(1)
        return cls_logits, height_pred

    def loss(self, cls_logits, height_pred, class_grid, height_grid):
        """
        Args:
            cls_logits (Tensor): (B, num_classes, H, W), from `forward`.
            height_pred (Tensor): (B, H, W), from `forward`.
            class_grid (Tensor): (B, H, W) long, IGNORE_CLASS where
                unsupervised.
            height_grid (Tensor): (B, H, W) float, value undefined (e.g. NaN)
                wherever class_grid == IGNORE_CLASS.

        Returns:
            dict: loss_occ_cls, loss_occ_height.
        """
        class_grid = class_grid.long()
        valid = class_grid != IGNORE_CLASS
        num_valid = valid.sum().clamp(min=1)

        loss_cls = self.loss_cls(cls_logits, class_grid, avg_factor=num_valid)

        if valid.any():
            loss_height = self.loss_height(height_pred[valid], height_grid[valid])
        else:
            # Keep height_pred in the autograd graph even with no valid cells
            # this step (e.g. an all-missing-GT batch), instead of skipping
            # the term and leaving its parameters without a gradient.
            loss_height = height_pred.sum() * 0.0

        return dict(loss_occ_cls=loss_cls, loss_occ_height=loss_height)

    def get_occ(self, cls_logits, height_pred):
        """Convenience for inference/visualization: argmax class + height.

        Returns:
            class_map (Tensor): (B, H, W) predicted class id.
            height_map (Tensor): (B, H, W) predicted height.
        """
        class_map = cls_logits.argmax(dim=1)
        return class_map, height_pred
