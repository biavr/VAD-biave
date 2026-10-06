"""Watches a training run's work_dir for new checkpoints and re-renders
occupancy predictions against a fixed set of samples on each one, so the
live dashboard can show "what the network currently predicts" updating
alongside the loss curves -- not just numbers.

Standalone process, same spirit as server.py and healthcheck.py: reads
files training already writes (checkpoints), touches nothing in the
training loop itself, and training is completely unaffected by whether
this is running or not.

The model is built once and reused across checkpoints (only the weights
get reloaded each time -- rebuilding the whole model/dataset per checkpoint
would be needlessly slow for something polling every few seconds), and the
same sample indices are tracked across checkpoints so the preview shows the
same scene improving over time, not a different random one each refresh.

Usage (separate terminal/pane, alongside training and the dashboard
server):
    python tools/live_dashboard/occ_watcher.py \\
        projects/configs/VAD/VAD_tiny_stage_1_with_occ.py \\
        --work-dir /workspace/logs/outputs_tiny_stage_1_with_occ \\
        --device cuda:1

Writes <work_dir>/occ_preview/latest.png (what server.py serves) plus a
timestamped copy per checkpoint for a history.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis_tools'))
from visualize_occupancy import (  # noqa: E402
    build_class_colormap, build_dataset, build_model, find_occ_gt_indices, render_sample,
)
from mmengine.runner import load_checkpoint  # noqa: E402


def find_latest_checkpoint(work_dir: Path):
    """Search work_dir recursively, not just the current run folder: the
    CheckpointHook has been observed saving directly into work_dir rather
    than the timestamped run subfolder it logs into (the same quirk
    server.py's find_checkpoints() already works around)."""
    ckpts = list(work_dir.rglob('iter_*.pth')) + list(work_dir.rglob('epoch_*.pth'))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--poll-secs', type=float, default=10)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    cfg = Config.fromfile(args.config)

    print('Building dataset and model once -- this takes a minute, then updates are cheap...')
    dataset = build_dataset(cfg)
    indices = find_occ_gt_indices(dataset, args.num_samples, args.seed)
    colormap = build_class_colormap()
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
                    model = build_model(cfg, str(ckpt_path), args.device)
                else:
                    load_checkpoint(model, str(ckpt_path), map_location='cpu')
                    model.to(args.device)
                    model.eval()

                preview_dir = work_dir / 'occ_preview'
                preview_dir.mkdir(parents=True, exist_ok=True)
                for i, (idx, token, sample) in enumerate(samples):
                    img = sample['inputs']['img'].unsqueeze(0)
                    cls_logits, height_pred = model.predict_occ(img, [sample['data_samples']])
                    out_path = preview_dir / f'sample_{i}_{token}.png'
                    render_sample(sample, cls_logits, height_pred, colormap, out_path, token)
                    shutil.copyfile(out_path, preview_dir / f'sample_{i}_latest.png')
                print(f'  Rendered {len(samples)} preview(s) from {ckpt_path.name}')
                last_ckpt_path = ckpt_path
            except Exception as e:
                print(f'  Failed to render from {ckpt_path}: {e!r}')
        time.sleep(args.poll_secs)


if __name__ == '__main__':
    main()
