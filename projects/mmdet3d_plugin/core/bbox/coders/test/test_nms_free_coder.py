"""Unit tests for CustomNMSFreeCoder (projects/mmdet3d_plugin/core/bbox/
coders/fut_nms_free_coder.py) -- converts raw (cls_scores, bbox_preds) into
final (boxes, scores, labels) for evaluation/submission. The
labels = indexs % num_classes / bbox_index = indexs // num_classes trick is
a classic site for a transposed-dimension or off-by-one bug: if either
direction is swapped, every decoded label would be wrong while shapes and
loss values stay completely unremarkable.
"""
import torch

from projects.mmdet3d_plugin.core.bbox.coders.fut_nms_free_coder import CustomNMSFreeCoder

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
NUM_CLASSES = 10


def build_coder(**kwargs):
    defaults = dict(
        pc_range=PC_RANGE,
        post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
        max_num=100,
        num_classes=NUM_CLASSES,
    )
    defaults.update(kwargs)
    return CustomNMSFreeCoder(**defaults)


def test_decode_single_recovers_correct_label_for_each_query():
    """One query per class, each with an unambiguous single-class spike in
    its logits. If the labels = indexs % num_classes decoding is correct,
    each decoded box's label must match the class whose logit was spiked
    for that query -- in whatever score-sorted order they come out.

    NOTE: bbox_preds reaching this coder are already real-world units for
    cx/cy/cz -- that rescaling happens upstream in VAD_head.forward()
    (tmp[...,0:1] * (pc_range[3]-pc_range[0]) + pc_range[0]), not here.
    denormalize_bbox (used internally by decode_single) passes cx/cy/cz
    through unchanged, confirmed in test_bbox_util.py. So distinguishing
    cx values here should be given directly in real-world units, not
    normalized [0,1]."""
    coder = build_coder(max_num=NUM_CLASSES)
    num_query = NUM_CLASSES
    cls_scores = torch.full((num_query, NUM_CLASSES), -10.0)
    expected_label_for_query = {}
    for q in range(num_query):
        true_class = (q * 3 + 1) % NUM_CLASSES  # some non-trivial permutation
        cls_scores[q, true_class] = 10.0 - q * 0.01  # distinct scores, no ties
        expected_label_for_query[q] = true_class

    bbox_preds = torch.zeros(num_query, 10)
    bbox_preds[:, 0] = torch.linspace(-14.0, 14.0, num_query)  # distinguishable real-world cx per query
    traj_preds = torch.zeros(num_query, 6, 2)  # fut_mode=6, fut_ts placeholder dims unused by this test

    result = coder.decode_single(cls_scores, bbox_preds, traj_preds)
    # all cx values are within post_center_range's x bounds [-20, 20], so
    # every query should survive the filter.
    assert len(result['scores']) == num_query

    # For each surviving box, recover which original query it came from via
    # its cx (we gave each query a distinct cx), and check its decoded label
    # matches that query's spiked class.
    for box_cx, label in zip(result['bboxes'][:, 0], result['labels']):
        q = torch.argmin((bbox_preds[:, 0] - box_cx).abs()).item()
        assert label.item() == expected_label_for_query[q], (
            f"query {q} spiked class {expected_label_for_query[q]} but decoded label was {label.item()}"
        )


def test_decode_single_respects_post_center_range():
    """bbox_preds' cx/cy/cz are already real-world units by this point (see
    note in the test above) -- post_center_range filters directly against
    those, with no further scaling."""
    coder = build_coder(max_num=5, post_center_range=[-1, -1, -1, 1, 1, 1])
    cls_scores = torch.zeros(3, NUM_CLASSES)
    cls_scores[:, 0] = 10.0  # all confidently "class 0"
    bbox_preds = torch.zeros(3, 10)
    bbox_preds[0, 0] = 0.5    # inside [-1, 1]
    bbox_preds[1, 0] = 15.0   # outside [-1, 1]
    bbox_preds[2, 0] = -15.0  # outside [-1, 1]
    traj_preds = torch.zeros(3, 6, 2)

    result = coder.decode_single(cls_scores, bbox_preds, traj_preds)
    assert len(result['scores']) == 1, "only the in-range box should survive post_center_range filtering"


def test_decode_topk_picks_highest_scores_first():
    """Scores should come out sorted descending (NMS-free top-k selection)."""
    coder = build_coder(max_num=4)
    cls_scores = torch.zeros(6, NUM_CLASSES)
    # distinct max-score per query, deliberately out of order
    for q, val in enumerate([0.1, 0.9, 0.3, 0.7, 0.2, 0.5]):
        cls_scores[q, 0] = val * 10  # logit scale, order preserved by sigmoid (monotonic)
    bbox_preds = torch.zeros(6, 10)
    traj_preds = torch.zeros(6, 6, 2)

    result = coder.decode_single(cls_scores, bbox_preds, traj_preds)
    scores_sorted = result['scores']
    assert len(scores_sorted) == 4
    assert torch.all(scores_sorted[:-1] >= scores_sorted[1:]), "decoded scores must be sorted descending"
