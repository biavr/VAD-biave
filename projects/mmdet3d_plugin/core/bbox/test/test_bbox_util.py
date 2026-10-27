"""Unit tests for projects/mmdet3d_plugin/core/bbox/util.py
(normalize_bbox / denormalize_bbox) -- the encode/decode pair used
throughout training (loss target construction) and inference (coder
decode). A bug here would silently corrupt every box regression target.
"""
import numpy as np
import torch

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]  # unused by the functions themselves, but passed for signature parity


def test_normalize_denormalize_roundtrip_no_velocity():
    torch.manual_seed(0)
    n = 20
    cx = torch.empty(n).uniform_(-15, 15)
    cy = torch.empty(n).uniform_(-30, 30)
    cz = torch.empty(n).uniform_(-2, 2)
    w = torch.empty(n).uniform_(0.2, 5.0)
    l = torch.empty(n).uniform_(0.2, 10.0)
    h = torch.empty(n).uniform_(0.5, 4.0)
    rot = torch.empty(n).uniform_(-np.pi, np.pi)
    raw = torch.stack([cx, cy, cz, w, l, h, rot], dim=-1)

    normalized = normalize_bbox(raw, PC_RANGE)
    assert normalized.shape == (n, 8), f"expected 8 normalized channels, got {normalized.shape}"

    recovered = denormalize_bbox(normalized, PC_RANGE)
    assert recovered.shape == (n, 7)

    # cx, cy, cz, w, l, h should round-trip exactly (log/exp, direct copy)
    torch.testing.assert_close(recovered[:, 0], raw[:, 0], msg="cx mismatch")
    torch.testing.assert_close(recovered[:, 1], raw[:, 1], msg="cy mismatch")
    torch.testing.assert_close(recovered[:, 2], raw[:, 2], msg="cz mismatch")
    torch.testing.assert_close(recovered[:, 3], raw[:, 3], msg="w mismatch")
    torch.testing.assert_close(recovered[:, 4], raw[:, 4], msg="l mismatch")
    torch.testing.assert_close(recovered[:, 5], raw[:, 5], msg="h mismatch")
    # rotation round-trips modulo the sin/cos -> atan2 range; since rot was
    # drawn from (-pi, pi], atan2 should recover it directly.
    torch.testing.assert_close(recovered[:, 6], raw[:, 6], atol=1e-5, rtol=1e-4, msg="rotation mismatch")


def test_normalize_denormalize_roundtrip_with_velocity():
    torch.manual_seed(1)
    n = 15
    raw = torch.cat([
        torch.empty(n, 1).uniform_(-15, 15),   # cx
        torch.empty(n, 1).uniform_(-30, 30),   # cy
        torch.empty(n, 1).uniform_(-2, 2),     # cz
        torch.empty(n, 1).uniform_(0.2, 5.0),  # w
        torch.empty(n, 1).uniform_(0.2, 10.0), # l
        torch.empty(n, 1).uniform_(0.5, 4.0),  # h
        torch.empty(n, 1).uniform_(-np.pi, np.pi),  # rot
        torch.empty(n, 1).uniform_(-10, 10),   # vx
        torch.empty(n, 1).uniform_(-10, 10),   # vy
    ], dim=-1)

    normalized = normalize_bbox(raw, PC_RANGE)
    assert normalized.shape == (n, 10)
    recovered = denormalize_bbox(normalized, PC_RANGE)
    assert recovered.shape == (n, 9)

    for i, name in enumerate(['cx', 'cy', 'cz', 'w', 'l', 'h']):
        torch.testing.assert_close(recovered[:, i], raw[:, i], msg=f"{name} mismatch")
    torch.testing.assert_close(recovered[:, 6], raw[:, 6], atol=1e-5, rtol=1e-4, msg="rotation mismatch")
    torch.testing.assert_close(recovered[:, 7], raw[:, 7], msg="vx mismatch")
    torch.testing.assert_close(recovered[:, 8], raw[:, 8], msg="vy mismatch")


def test_normalize_bbox_index_mapping_is_not_a_silent_permutation_bug():
    """Catches the specific bug class of indices being swapped (e.g. w/l or
    cz/h transposed) by using distinguishable, non-symmetric values for
    every channel and checking each lands exactly where documented."""
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]])  # cx,cy,cz,w,l,h,rot=0
    normalized = normalize_bbox(raw, PC_RANGE)
    # documented normalized layout: cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)
    assert normalized[0, 0].item() == 1.0  # cx
    assert normalized[0, 1].item() == 2.0  # cy
    assert abs(normalized[0, 2].item() - np.log(4.0)) < 1e-5  # log(w)
    assert abs(normalized[0, 3].item() - np.log(5.0)) < 1e-5  # log(l)
    assert normalized[0, 4].item() == 3.0  # cz
    assert abs(normalized[0, 5].item() - np.log(6.0)) < 1e-5  # log(h)
    assert abs(normalized[0, 6].item() - 0.0) < 1e-6  # sin(0)
    assert abs(normalized[0, 7].item() - 1.0) < 1e-6  # cos(0)
