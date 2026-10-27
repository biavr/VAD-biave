"""Regression test for a real bug in VADHead.get_best_fut_preds /
get_traj_cls_target (trajectory/motion loss path -- flagged as "not yet
tested" in FINDINGS.md from the earlier investigation phase).

Both functions attempt to zero out NaN entries in the per-mode distance
tensor via `dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0` --
but `nan * 0 == nan` in IEEE 754, so this is a no-op: NaN values survive
unchanged. The immediate downstream call is `torch.argmin(dist, dim=-1)`,
and torch.argmin PREFERS NaN over any finite value (confirmed directly:
`torch.argmin(torch.tensor([5.0, float('nan'), 2.0]))` returns 1, the NaN
index, not 2, the true minimum). So whenever any trajectory mode's
cumulative distance comes out NaN (plausible from a degenerate/fully-masked
future), that mode gets selected as "best" / used as the classification
target, instead of being excluded as invalid -- the opposite of what the
(failed) NaN-zeroing was presumably meant to achieve.

This exercises both functions with a synthetic case where one mode is
NaN and another is a clear, finite, correct minimum -- demonstrating the
bug, then (after the fix) confirming the NaN mode is correctly skipped.
"""
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


def test_get_best_fut_preds_ignores_nan_mode():
    head = build_head()
    fut_ts = head.fut_ts
    fut_mode = head.fut_mode
    num_box = 1

    # mode 0: NaN cumulative distance (e.g. from a degenerate prediction)
    # mode 1: exactly matches the GT trajectory -> true distance 0
    # mode 2: clearly worse than mode 1 but still finite
    traj_preds = torch.zeros(num_box, fut_mode, fut_ts, 2)
    traj_preds[0, 0, 0, 0] = float('nan')
    gt_traj = torch.arange(fut_ts, dtype=torch.float32).unsqueeze(-1).repeat(1, 2) * 0.5
    traj_preds[0, 1] = gt_traj  # perfect match -> dist 0
    traj_preds[0, 2] = gt_traj + 5.0  # clearly worse, but finite

    traj_targets = gt_traj.unsqueeze(0)  # (num_box, fut_ts, 2)
    gt_fut_masks = torch.ones(num_box, fut_ts)

    best = head.get_best_fut_preds(traj_preds, traj_targets, gt_fut_masks)

    # The correctly-matching mode (1) should be selected, NOT the NaN
    # mode (0). If this fails, best will be all-NaN (mode 0 was picked).
    assert torch.isfinite(best).all(), (
        f"get_best_fut_preds selected the NaN mode instead of the correct "
        f"finite match -- best contains NaN: {best}")
    torch.testing.assert_close(
        best.reshape(fut_ts, 2), gt_traj,
        msg="get_best_fut_preds did not select the perfectly-matching mode")


def test_get_traj_cls_target_ignores_nan_mode():
    head = build_head()
    fut_ts = head.fut_ts
    fut_mode = head.fut_mode
    num_box = 1

    traj_preds = torch.zeros(num_box, fut_mode, fut_ts, 2)
    traj_preds[0, 0, 0, 0] = float('nan')  # mode 0: NaN
    gt_traj = torch.arange(fut_ts, dtype=torch.float32).unsqueeze(-1).repeat(1, 2) * 0.5
    traj_preds[0, 1] = gt_traj  # mode 1: perfect match
    traj_preds[0, 2] = gt_traj + 5.0  # mode 2: worse but finite

    traj_targets = gt_traj.unsqueeze(0)
    gt_fut_masks = torch.ones(num_box, fut_ts)
    neg_inds = torch.zeros(num_box, dtype=torch.bool)

    traj_labels = head.get_traj_cls_target(traj_preds, traj_targets, gt_fut_masks, neg_inds)

    assert traj_labels[0].item() == 1, (
        f"expected mode 1 (perfect match) to be the classification target, "
        f"got mode {traj_labels[0].item()} -- the NaN mode (0) was likely "
        f"preferred instead")
