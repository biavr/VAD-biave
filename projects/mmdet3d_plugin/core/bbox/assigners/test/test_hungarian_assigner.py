"""Unit tests for HungarianAssigner3D (projects/mmdet3d_plugin/core/bbox/
assigners/hungarian_assigner_3d.py) -- the component responsible for
deciding which predicted query gets supervised against which ground-truth
box during training. A bug here (wrong cost scale, wrong matching, wrong
label assignment) would silently corrupt the detection training target
while leaving the loss computation itself looking perfectly correct --
exactly the kind of bug that would produce "loss decreases, mAP stays at
0" without an obvious stack trace anywhere.
"""
import torch

from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]


def build_assigner():
    return HungarianAssigner3D(
        cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
        reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.25),
        iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
        pc_range=PC_RANGE,
    )


def make_box(cx, cy, cz=0.0, w=2.0, l=4.0, h=1.5, rot=0.0, vx=0.0, vy=0.0):
    return [cx, cy, cz, w, l, h, rot, vx, vy]


def test_assigner_matches_each_prediction_to_its_nearest_gt():
    """3 well-separated GT boxes, 3 predictions each placed right on top of
    a *different* one of them (in a shuffled order) plus matching class
    scores. A correct assigner must recover the obviously-correct
    one-to-one pairing regardless of the shuffle."""
    assigner = build_assigner()
    gt_bboxes = torch.tensor([
        make_box(-10.0, -20.0),
        make_box(0.0, 0.0),
        make_box(10.0, 20.0),
    ])
    gt_labels = torch.tensor([0, 1, 2])

    # Predictions in a DIFFERENT order than the GTs, each sitting almost
    # exactly on one GT, with a strong, correct, confident class score.
    num_classes = 10
    bbox_pred = torch.stack([
        torch.tensor(make_box(10.05, 20.1)),   # near gt[2]
        torch.tensor(make_box(-9.9, -19.8)),   # near gt[0]
        torch.tensor(make_box(0.05, -0.1)),    # near gt[1]
    ] + [torch.tensor(make_box(100.0, 100.0)) for _ in range(5)])  # far-away distractors
    cls_pred = torch.full((8, num_classes), -5.0)  # strong "nothing" logits by default
    cls_pred[0, 2] = 5.0  # confidently predicts class 2 (matches gt[2])
    cls_pred[1, 0] = 5.0  # confidently predicts class 0 (matches gt[0])
    cls_pred[2, 1] = 5.0  # confidently predicts class 1 (matches gt[1])

    result = assigner.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
    assigned = result.gt_inds  # 0 = background, else 1-based gt index

    assert assigned[0].item() == 3, "query 0 (near gt[2]) should be assigned to gt index 3 (1-based)"
    assert assigned[1].item() == 1, "query 1 (near gt[0]) should be assigned to gt index 1 (1-based)"
    assert assigned[2].item() == 2, "query 2 (near gt[1]) should be assigned to gt index 2 (1-based)"
    assert (assigned[3:] == 0).all(), "far-away distractor queries should all be background"

    assert result.labels[0].item() == 2
    assert result.labels[1].item() == 0
    assert result.labels[2].item() == 1


def test_assigner_handles_zero_gt_without_crashing():
    assigner = build_assigner()
    bbox_pred = torch.stack([torch.tensor(make_box(0.0, 0.0)) for _ in range(4)])
    cls_pred = torch.zeros(4, 10)
    gt_bboxes = torch.zeros(0, 9)
    gt_labels = torch.zeros(0, dtype=torch.long)

    result = assigner.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
    assert (result.gt_inds == 0).all(), "with no GT, every query must be background"


def test_assigner_handles_nan_cost_gracefully():
    """A NaN anywhere in bbox_pred (e.g. from an upstream numerical issue)
    must not crash the Hungarian solver -- the existing code replaces NaN/Inf
    costs with a large finite sentinel; verify that path actually runs."""
    assigner = build_assigner()
    bbox_pred = torch.stack([torch.tensor(make_box(float('nan'), 0.0))] +
                             [torch.tensor(make_box(0.0, 0.0)) for _ in range(3)])
    cls_pred = torch.zeros(4, 10)
    gt_bboxes = torch.tensor([make_box(0.0, 0.0)])
    gt_labels = torch.tensor([0])

    result = assigner.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
    assert result.gt_inds.shape == (4,)
    assert torch.isfinite(result.gt_inds.float()).all()
