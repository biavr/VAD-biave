"""True end-to-end sanity check: one real training step (forward_train +
backward) on one real mini-dataset sample, checking:
  - no NaN/Inf in any loss term
  - gradients reach both the classification and regression branches, with
    comparable magnitude (catches a silently-frozen/disconnected branch)
This is deliberately not about whether the *values* are good -- just that
the whole pipeline is wired together correctly end to end on real data.

Needs the nuScenes-mini dataset plus a path-fixed temporal-info pkl that
isn't shipped in this repo (generated once during an earlier investigation
via a one-off path-fixup script, since the mini pkls shipped with
v1.0-mini had Windows-style paths baked in from wherever they were
generated). If that external config/data isn't present -- e.g. on a fresh
clone, or if /workspace/logs/investigation/ has been cleaned up -- this
test skips rather than failing the suite.
"""
import sys
from pathlib import Path

import pytest
import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config

sys.path.insert(0, '/workspace/VAD/tools/analysis_tools')
from visualize_occupancy import build_dataset  # noqa: E402

CONFIG = '/workspace/logs/investigation/VAD_mini_unittest.py'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_one_real_training_step_has_finite_losses_and_live_gradients():
    if not Path(CONFIG).exists():
        pytest.skip(f"mini-dataset config not found at {CONFIG} -- see this file's "
                    f"docstring for how to regenerate it")
    cfg = Config.fromfile(CONFIG)
    dataset = build_dataset(cfg, require_occ=False)
    model = MODELS.build(cfg.model).to(DEVICE)
    model.train()

    idx = 0
    while idx < len(dataset):
        sample = dataset[idx]
        ds = sample['data_samples']
        if hasattr(ds, 'gt_instances_3d') and len(ds.gt_instances_3d.bboxes_3d) >= 3:
            break
        idx += 1
    assert idx < len(dataset), "no mini sample with >=3 GT boxes found"

    img = sample['inputs']['img'].unsqueeze(0).to(DEVICE).float()
    ds = ds.to(DEVICE)

    model.zero_grad()
    losses = model.forward_train(img=img, data_samples=[ds])

    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), f"{name} is not finite: {value}"

    total_loss = sum(v for k, v in losses.items() if 'loss' in k and isinstance(v, torch.Tensor))
    assert torch.isfinite(total_loss), "summed total loss is not finite"
    total_loss.backward()

    def branch_grad_norm(branches):
        total = 0.0
        for branch in branches:
            for p in branch.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
        return total ** 0.5

    cls_grad_norm = branch_grad_norm(model.pts_bbox_head.cls_branches)
    reg_grad_norm = branch_grad_norm(model.pts_bbox_head.reg_branches)

    assert cls_grad_norm > 0, "classification branch received zero gradient -- likely disconnected from the loss"
    assert reg_grad_norm > 0, "regression branch received zero gradient -- likely disconnected from the loss"
    # Not a tight bound -- just catches a branch being orders of magnitude
    # weaker than the other (e.g. a detached tensor silently short-circuiting
    # most of the gradient path for one branch only).
    ratio = max(cls_grad_norm, reg_grad_norm) / max(min(cls_grad_norm, reg_grad_norm), 1e-12)
    assert ratio < 1000, (
        f"cls/reg gradient norms differ by {ratio:.1f}x (cls={cls_grad_norm:.6f}, reg={reg_grad_norm:.6f}) "
        "-- suspiciously large imbalance, check for a disconnected branch"
    )
