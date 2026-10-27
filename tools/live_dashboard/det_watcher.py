"""Watches a training run's work_dir for new checkpoints and re-renders
predicted + GT 3D detection boxes (projected onto all 6 camera views) for a
fixed set of samples on each one, the same spirit as occ_watcher.py but for
detection instead of occupancy -- so the dashboard shows "what boxes does
the model currently predict", not just loss curves.

Standalone process: builds the dataset + model once, then on each new
checkpoint just reloads weights and re-renders. Needs its own GPU, separate
from whatever GPU(s) training itself uses.

Usage:
    python tools/live_dashboard/det_watcher.py \\
        <config> --work-dir <work_dir> --device cuda:0
"""
import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_tools'))
from visualize_occupancy import CAM_ORDER, build_dataset, build_model, denormalize_for_display  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
SCORE_THRESH = 0.2


def find_latest_checkpoint(work_dir: Path):
    """Same work_dir-wide search as occ_watcher.py's -- CheckpointHook has
    been observed saving to either the work_dir root or the timestamped run
    subfolder depending on the run, so search both via rglob."""
    ckpts = list(work_dir.rglob('iter_*.pth')) + list(work_dir.rglob('epoch_*.pth'))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def project_corners(corners, lidar2img):
    """corners: (8, 3) in lidar frame. lidar2img: (4, 4). Returns (8, 2) pixel coords, (8,) depth."""
    hom = np.concatenate([corners, np.ones((8, 1))], axis=1)
    proj = hom @ lidar2img.T
    depth = proj[:, 2]
    pts_2d = proj[:, :2] / np.clip(depth[:, None], 1e-4, None)
    return pts_2d, depth


def draw_box(ax, corners_2d, depth, color, linewidth=1.5):
    if (depth <= 0).all():
        return
    for i, j in EDGES:
        if depth[i] <= 0 or depth[j] <= 0:
            continue
        ax.plot([corners_2d[i, 0], corners_2d[j, 0]], [corners_2d[i, 1], corners_2d[j, 1]],
                color=color, linewidth=linewidth)


def render_sample(model, sample, token, out_path, device):
    img_queue = sample['inputs']['img']
    len_queue = img_queue.shape[0]
    img_metas_curr = sample['data_samples'].metainfo['img_metas'][len_queue - 1]
    lidar2img = np.asarray(img_metas_curr['lidar2img'])
    img_curr = img_queue[-1]

    # model's 'predict' mode is built for test-time samples, where
    # metainfo['img_metas'] is a single flat dict (no temporal queue) -- but
    # we're reusing the *training* dataset wrapper (for its queue/GT
    # handling), whose samples carry img_metas as a list indexed by frame.
    # Build a single-frame view: current-frame image only, current-frame
    # meta promoted to the flat shape 'predict' expects.
    img_single = img_curr.unsqueeze(0).to(device).float()
    ds = copy.deepcopy(sample['data_samples'])
    ds.set_metainfo({'img_metas': img_metas_curr})
    ds = ds.to(device)
    with torch.no_grad():
        results = model(inputs=img_single, data_samples=[ds], mode='predict')
    pred = results[0]['pred_instances_3d']
    boxes_3d, scores_3d = pred['bboxes_3d'], pred['scores_3d']
    keep = scores_3d > SCORE_THRESH
    pred_corners = boxes_3d.corners[keep].cpu().numpy() if keep.any() else np.zeros((0, 8, 3))

    # Unlike the test dataset wrapper (which nests GT inside img_metas), the
    # training wrapper we reuse here (for its queue/occ-GT handling) carries
    # real detection GT at data_samples.gt_instances_3d instead -- confirmed
    # during the earlier GT-wiring investigation.
    gt_src = sample['data_samples']
    gt_boxes_3d = gt_src.gt_instances_3d.bboxes_3d if hasattr(gt_src, 'gt_instances_3d') else None
    gt_corners = gt_boxes_3d.corners.numpy() if gt_boxes_3d is not None and len(gt_boxes_3d) > 0 else np.zeros((0, 8, 3))

    num_cams = img_curr.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.flatten()
    for cam_i in range(num_cams):
        ax = axes[cam_i]
        img = denormalize_for_display(img_curr[cam_i])
        ax.imshow(img)
        ax.set_title(CAM_ORDER[cam_i], fontsize=9)
        ax.axis('off')
        for box_i in range(len(gt_corners)):
            pts_2d, depth = project_corners(gt_corners[box_i], lidar2img[cam_i])
            draw_box(ax, pts_2d, depth, color='red', linewidth=2)
        for box_i in range(len(pred_corners)):
            pts_2d, depth = project_corners(pred_corners[box_i], lidar2img[cam_i])
            draw_box(ax, pts_2d, depth, color='lime')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    for extra in range(num_cams, len(axes)):
        axes[extra].axis('off')

    fig.suptitle(f'token={token}  (red=GT {len(gt_corners)}, green=pred>{SCORE_THRESH} {len(pred_corners)})',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--poll-secs', type=float, default=10)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from mmengine.config import Config
    work_dir = Path(args.work_dir)
    cfg = Config.fromfile(args.config)

    print('Building dataset and model once -- this takes a minute, then updates are cheap...')
    dataset = build_dataset(cfg, require_occ=False)
    rng = np.random.RandomState(args.seed)
    # Unlike occupancy GT (sparse), detection GT exists for virtually every
    # sample -- just pick samples with a reasonable number of boxes so the
    # preview is informative rather than an empty scene.
    candidates = [i for i, info in enumerate(dataset.data_infos) if len(info.get('gt_boxes', [])) >= 5]
    indices = rng.choice(candidates, size=min(args.num_samples, len(candidates)), replace=False)
    samples = [(idx, dataset.data_infos[idx]['token'], dataset[idx]) for idx in indices]
    print(f'Tracking {len(samples)} fixed sample(s): {[t for _, t, _ in samples]}')

    model = None
    last_ckpt_path = None

    print(f'Watching {work_dir} for new checkpoints every {args.poll_secs}s. Ctrl+C to stop.')
    while True:
        ckpt_path = find_latest_checkpoint(work_dir)
        if ckpt_path is not None and ckpt_path != last_ckpt_path:
            print(f'New checkpoint: {ckpt_path}')
            try:
                if model is None:
                    model = build_model(cfg, str(ckpt_path), args.device, require_occ=False)
                else:
                    load_checkpoint(model, str(ckpt_path), map_location='cpu')
                    model.to(args.device)
                    model.eval()

                preview_dir = work_dir / 'det_preview'
                preview_dir.mkdir(parents=True, exist_ok=True)
                for i, (idx, token, sample) in enumerate(samples):
                    out_path = preview_dir / f'sample_{i}_{token}.png'
                    render_sample(model, sample, token, out_path, args.device)
                    shutil.copyfile(out_path, preview_dir / f'sample_{i}_latest.png')
                print(f'  Rendered {len(samples)} preview(s) from {ckpt_path.name}')
                last_ckpt_path = ckpt_path
            except Exception as e:
                print(f'  Failed to render from {ckpt_path}: {e!r}')
        time.sleep(args.poll_secs)


if __name__ == '__main__':
    main()
