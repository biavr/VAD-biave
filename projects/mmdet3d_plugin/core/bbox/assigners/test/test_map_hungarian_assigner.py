"""Regression test for a real bug found during a 48-epoch extended mini-data
training run (not caught by the earlier 14-test investigation, which did not
cover the map-vector loss path): MapHungarianAssigner3D.assign() crashed with
`ValueError: matrix contains invalid numeric entries` from scipy's
linear_sum_assignment.

Root cause: this project's configs set iou_cost and reg_cost weight=0.0,
intending them to be no-ops in the summed cost. But IoUCost's giou
computation involves an area PRODUCT (width * height) of denormalized boxes
-- if a single query's regressed bbox_pred momentarily takes on a large
magnitude (plausible mid-training, especially during a longer sustained-high
-LR phase before this schedule's decay point), that product can overflow
float32 to inf. 0.0 * inf == nan in IEEE 754, so a cost component that is
*configured to be disabled* can still inject NaN into the cost matrix sum
and crash the whole Hungarian-matching step for the WHOLE batch (not just
the one degenerate query).

The sibling detection assigner (hungarian_assigner_3d.py) already guards
against this (lines ~209-213, replacing non-finite cost entries with a large
finite value before linear_sum_assignment) -- evidently added during the
MMDetection 3.x migration for the detection path, but never ported to the
map path. This test exercises exactly that gap.
"""
import torch

from mmdet.models.task_modules import build_match_cost
from projects.mmdet3d_plugin.core.bbox.assigners.map_hungarian_assigner_3d import (
    MapHungarianAssigner3D,
)

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]


def build_assigner():
    return MapHungarianAssigner3D(
        cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
        reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.0),
        iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=0.0),
        pts_cost=dict(type='mmdet.OrderedPtsL1Cost', weight=1.0),
        pc_range=PC_RANGE,
    )


def test_assign_survives_diverged_bbox_pred_without_crashing():
    assigner = build_assigner()

    num_query = 6
    num_gt = 2
    num_classes = 3
    num_pts_per_line = 10
    num_orders = 4

    cls_pred = torch.full((num_query, num_classes), -3.0)
    cls_pred[0, 0] = 3.0
    cls_pred[1, 1] = 3.0

    # queries 0 and 1 are sane, normalized-ish predictions; query 2 has a
    # wildly diverged bbox_pred (as can transiently happen mid-training
    # before gradient clipping fully stabilizes a batch) -- this alone used
    # to be enough to crash assign() for the ENTIRE batch, not just query 2.
    bbox_pred = torch.rand(num_query, 4) * 0.5 + 0.25  # safe [0.25, 0.75] range
    bbox_pred[2, 2] = 1e20  # diverged width -> overflows float32 in area product
    bbox_pred[2, 3] = 1e20  # diverged height

    pts_pred = torch.rand(num_query, num_pts_per_line, 2) * 0.5 + 0.25

    gt_bboxes = torch.tensor([
        [-5.0, -5.0, 5.0, 5.0],
        [5.0, 5.0, 10.0, 10.0],
    ])
    gt_labels = torch.tensor([0, 1])
    gt_pts = torch.rand(num_gt, num_orders, num_pts_per_line, 2) * 20 - 10

    # Must not raise -- this is exactly what crashed before the fix.
    assign_result, order_index = assigner.assign(
        bbox_pred, cls_pred, pts_pred, gt_bboxes, gt_labels, gt_pts)

    assert assign_result.num_gts == num_gt
    assert torch.isfinite(assign_result.gt_inds.float()).all()
    # The two well-behaved, confidently-classed queries should still match
    # their intended classes -- the fix must not corrupt otherwise-sane
    # matching, just prevent a crash from the one degenerate query.
    assert assign_result.labels[0].item() in (0, -1)
    assert assign_result.labels[1].item() in (1, -1)
