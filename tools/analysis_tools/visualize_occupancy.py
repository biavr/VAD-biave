"""Qualitative occupancy-head check: predicted vs. reference BEV grid.

Nothing in the repo already does this -- tools/test_new.py and
tools/analysis_tools/visualization*.py only evaluate/render detection boxes
via the stock nuScenes devkit, which has no notion of this project's 2.5D
occupancy head at all. This is a small, standalone script instead of a
change to the training/test pipeline: it loads a checkpoint directly, runs
VAD.predict_occ() (added alongside this script) on a handful of samples,
and renders each one's camera views + GT/predicted occupancy side by side.

Only samples that actually have ground-truth occupancy data are picked
(tools/data_converter/occupancy_gt_generator.py's coverage is still partial
pending a full lidarseg download -- see its own completeness caveat), so
what's rendered is always a real comparison, never a placeholder.

Runs against the *train* split: val_dataloader is currently disabled in
these configs (the stock NuScenesMetric evaluator doesn't understand this
project's custom outputs), so train is the only dataset wired up to attach
occupancy ground truth today.

Usage:
    python tools/analysis_tools/visualize_occupancy.py \\
        projects/configs/VAD/VAD_tiny_stage_1_with_occ.py \\
        /workspace/logs/outputs_tiny_stage_1_with_occ/20261005_063425/epoch_3.pth \\
        --out-dir /workspace/logs/occ_viz --num-samples 8
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import DATASETS
from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint
from nuscenes.utils.color_map import get_colormap

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../VAD
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_CONVERTER_DIR = REPO_ROOT / 'tools' / 'data_converter'
if str(DATA_CONVERTER_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CONVERTER_DIR))

import projects.mmdet3d_plugin  # noqa: F401,E402 -- registers VAD, VADOccHead, the dataset, etc.
from projects.mmdet3d_plugin.VAD.occupancy_head import IGNORE_CLASS  # noqa: E402
from occupancy_gt_generator import LIDARSEG_IDX2NAME  # noqa: E402

CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def build_class_colormap():
    """(33, 3) uint8 array: rows 0-31 are the official nuScenes-lidarseg
    colors (same ones occupancy_gt_generator.py's indices refer to), row 32
    is a neutral gray standing in for IGNORE_CLASS (remapped from 255, since
    that's out of range for a 33-row array)."""
    name_to_rgb = get_colormap()
    colors = np.zeros((33, 3), dtype=np.uint8)
    for idx, name in LIDARSEG_IDX2NAME.items():
        colors[idx] = name_to_rgb[name]
    colors[32] = (128, 128, 128)  # IGNORE_CLASS slot
    return colors


def class_grid_to_rgb(grid, colormap):
    grid = grid.copy()
    grid[grid == IGNORE_CLASS] = 32
    return colormap[grid.astype(np.int64)]


def find_occ_gt_indices(dataset, num_samples, seed):
    """Indices of samples that actually have occupancy ground truth, so
    every rendered comparison is against real data, not an all-ignored
    placeholder grid."""
    candidates = [i for i, info in enumerate(dataset.data_infos)
                 if info['token'] in dataset._occ_gt_tokens]
    if not candidates:
        raise RuntimeError(
            'No samples with occupancy GT found. Is --work-dir pointing at a '
            'dataset whose occ_gt_root has been populated by '
            'tools/data_converter/occupancy_gt_generator.py?')
    random.Random(seed).shuffle(candidates)
    return candidates[:num_samples]


def denormalize_for_display(img_chw):
    """Dataset samples are raw pixel-ish values (mean/std normalization
    happens later, inside the model's own data_preprocessor) -- just clip
    to a displayable range rather than undo a normalization that hasn't
    been applied yet."""
    img = img_chw.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 255).astype(np.uint8)


def render_sample(sample, cls_logits, height_pred, colormap, out_path, token):
    data_sample = sample['data_samples']
    img = sample['inputs']['img']  # (queue_length, num_cams, C, H, W)
    img_curr = img[-1]  # current frame

    gt_class = np.asarray(data_sample.metainfo['occ_class_grid'])
    gt_height = np.asarray(data_sample.metainfo['occ_height_grid'])
    pred_class = cls_logits.argmax(1)[0].cpu().numpy()
    pred_height = height_pred[0].cpu().numpy()

    img_metas = data_sample.metainfo.get('img_metas', {})
    cam_names = img_metas.get('cams', CAM_ORDER) if isinstance(img_metas, dict) else CAM_ORDER
    if not isinstance(cam_names, (list, tuple)) or len(cam_names) != img_curr.shape[0]:
        cam_names = CAM_ORDER[:img_curr.shape[0]]

    fig, axes = plt.subplots(3, max(4, img_curr.shape[0] // 2), figsize=(18, 11))
    for ax in axes.flat:
        ax.axis('off')

    n_cam_cols = axes.shape[1]
    for i in range(min(img_curr.shape[0], 2 * n_cam_cols)):
        row, col = divmod(i, n_cam_cols)
        axes[row, col].imshow(denormalize_for_display(img_curr[i]))
        axes[row, col].set_title(cam_names[i] if i < len(cam_names) else f'cam_{i}', fontsize=9)

    valid_mask = gt_class != IGNORE_CLASS
    height_vmin = gt_height[valid_mask].min() if valid_mask.any() else -2.0
    height_vmax = gt_height[valid_mask].max() if valid_mask.any() else 2.0

    ax = axes[2, 0]
    ax.imshow(class_grid_to_rgb(gt_class, colormap), origin='lower')
    ax.set_title(f'GT occupancy class ({int(valid_mask.sum())} labelled cells)', fontsize=9)
    ax.axis('off')

    ax = axes[2, 1]
    ax.imshow(class_grid_to_rgb(pred_class, colormap), origin='lower')
    ax.set_title('Predicted occupancy class', fontsize=9)
    ax.axis('off')

    gt_height_masked = np.where(valid_mask, gt_height, np.nan)
    ax = axes[2, 2]
    im = ax.imshow(gt_height_masked, origin='lower', cmap='viridis', vmin=height_vmin, vmax=height_vmax)
    ax.set_title('GT height (masked to labelled cells)', fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if n_cam_cols > 3:
        ax = axes[2, 3]
        im = ax.imshow(pred_height, origin='lower', cmap='viridis', vmin=height_vmin, vmax=height_vmax)
        ax.set_title('Predicted height (whole grid)', fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for col in range(4, n_cam_cols):
            axes[2, col].axis('off')

    fig.suptitle(f'token={token}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_dataset(cfg):
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    if getattr(dataset, 'occ_gt_root', None) is None:
        raise RuntimeError('This config has no occ_gt_root set -- nothing to compare against.')
    return dataset


def build_model(cfg, checkpoint_path, device):
    """Build once; call load_checkpoint again on the same model to pick up
    a newer checkpoint cheaply (used by the live dashboard's watcher, which
    re-renders on every new checkpoint without rebuilding the model)."""
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu')
    model.to(device)
    model.eval()
    if model.occ_head is None:
        raise RuntimeError('This config has no occ_head configured.')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', default='./occ_viz')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    model = build_model(cfg, args.checkpoint, args.device)

    colormap = build_class_colormap()
    indices = find_occ_gt_indices(dataset, args.num_samples, args.seed)
    print(f'Rendering {len(indices)} samples with real occupancy GT to {out_dir}')

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['inputs']['img'].unsqueeze(0)  # add batch dim
        cls_logits, height_pred = model.predict_occ(img, [sample['data_samples']])
        token = dataset.data_infos[idx]['token']
        out_path = out_dir / f'{i:03d}_{token}.png'
        render_sample(sample, cls_logits, height_pred, colormap, out_path, token)
        print(f'  [{i + 1}/{len(indices)}] {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
