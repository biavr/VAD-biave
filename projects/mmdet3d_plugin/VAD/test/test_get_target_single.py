"""Integration-style test for VADHead._get_target_single (the GT -> training
-target wiring that sits directly on top of HungarianAssigner3D) -- this is
the function with the most "MMDetection 3.x Compatibility Patch" rewriting
(InstanceData wrapping, PseudoSampler construction) found during the code
read-through, and a natural place for a migration-introduced bug: the
assigner itself tests correctly in isolation (test_hungarian_assigner.py),
but a bug in how its result gets turned into bbox_targets/labels here would
be just as damaging and wouldn't show up in that narrower test.

Builds the real VADHead from the project's own stage-1 config (fresh/random
weights -- this tests wiring logic, not learned behavior, so untrained
weights are fine and avoid needing a checkpoint or a dataset).
"""
import sys
from pathlib import Path

import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config

# .../projects/mmdet3d_plugin/VAD/test/this_file.py -> .../projects/configs/...
CONFIG = str(Path(__file__).resolve().parents[3] / 'configs' / 'VAD' / 'VAD_tiny_stage_1_updated.py')


def build_head():
    cfg = Config.fromfile(CONFIG)
    model = MODELS.build(cfg.model)
    return model.pts_bbox_head


def test_get_target_single_assigns_correct_gt_box_to_matched_query():
    head = build_head()
    num_query = 20
    num_classes = head.num_classes
    fut_ts = head.fut_ts

    cls_score = torch.full((num_query, num_classes), -5.0)
    bbox_pred = torch.zeros(num_query, 10)
    # two well-separated, confidently-classed predictions, rest neutral/background
    bbox_pred[0, 0] = -10.0  # cx (already real-world units at this point in the pipeline)
    bbox_pred[0, 1] = -20.0  # cy
    cls_score[0, 3] = 10.0
    bbox_pred[5, 0] = 10.0
    bbox_pred[5, 1] = 20.0
    cls_score[5, 7] = 10.0

    gt_bboxes = torch.tensor([
        [-10.0, -20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],  # matches query 0
        [10.0, 20.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],    # matches query 5
    ])
    gt_labels = torch.tensor([3, 7])
    gt_attr_labels = torch.zeros(2, fut_ts * 3 + 10)  # pad a bit beyond the minimum slice

    (labels, label_weights, bbox_targets, bbox_weights, traj_targets,
     traj_weights, gt_fut_masks, pos_inds, neg_inds) = head._get_target_single(
        cls_score, bbox_pred, gt_labels, gt_bboxes, gt_attr_labels)

    pos_inds_set = set(pos_inds.tolist())
    assert pos_inds_set == {0, 5}, f"expected queries 0 and 5 to be positive, got {pos_inds_set}"

    assert labels[0].item() == 3
    assert labels[5].item() == 7
    assert (labels[[i for i in range(num_query) if i not in (0, 5)]] == head.num_classes).all(), (
        "every non-matched query's label must be the background sentinel (num_classes)")

    # bbox_targets at the matched rows must be the ASSIGNED gt box, not a
    # scrambled/wrong one or the other query's gt.
    torch.testing.assert_close(bbox_targets[0], gt_bboxes[0], msg="query 0's bbox_targets != its assigned GT")
    torch.testing.assert_close(bbox_targets[5], gt_bboxes[1], msg="query 5's bbox_targets != its assigned GT")

    assert bbox_weights[0].sum().item() > 0
    assert bbox_weights[5].sum().item() > 0
    for i in range(num_query):
        if i not in (0, 5):
            assert bbox_weights[i].sum().item() == 0, f"query {i} should have zero bbox_weight (background)"


def test_get_target_single_handles_zero_gt():
    head = build_head()
    num_query = 10
    cls_score = torch.zeros(num_query, head.num_classes)
    bbox_pred = torch.zeros(num_query, 10)
    gt_bboxes = torch.zeros(0, 9)
    gt_labels = torch.zeros(0, dtype=torch.long)
    gt_attr_labels = torch.zeros(0, head.fut_ts * 3 + 10)

    (labels, label_weights, bbox_targets, bbox_weights, traj_targets,
     traj_weights, gt_fut_masks, pos_inds, neg_inds) = head._get_target_single(
        cls_score, bbox_pred, gt_labels, gt_bboxes, gt_attr_labels)

    assert len(pos_inds) == 0
    assert (labels == head.num_classes).all()
    assert bbox_weights.sum().item() == 0
