"""Log a chain of consecutive nuScenes samples -- camera images, GT
occupancy, and a live model's predicted occupancy -- onto a single Rerun
timeline, so you can scrub through a scene and compare prediction vs ground
truth frame by frame.

Unlike the `lidarseg`/`occupancy` loaders (which just read one file off
disk), this needs a trained model: it builds the dataset + model once (the
same way `tools/analysis_tools/visualize_occupancy.py` and
`tools/live_dashboard/occ_watcher.py` do) and runs a real forward pass per
frame. That's a different enough shape of tool that it isn't wired into the
`LOADERS` registry / `rerun <path>` plugin dispatch (those are for
"recognize this file, log it" cases) -- it's its own entry point instead,
following the same core.py building blocks (`log_occupancy_grid`,
`annotation_context`) so a GT box and a predicted box always look the same
except for which entity path they sit under.

Occupancy boxes are logged ego-centered every frame (the same local frame
`pc_range` is always defined in, exactly like `occ_watcher.py`'s single-frame
preview) rather than composed into one shared moving-vehicle trajectory --
simpler, and what you want for judging per-frame prediction quality.

Usage:
    python -m nuscenes_rerun.sequence \\
        projects/configs/VAD/VAD_tiny_stage_1_with_occ.py \\
        /workspace/logs/outputs_tiny_stage_1_with_occ/20261005_103643/epoch_3.pth \\
        --num-frames 20 --save sequence.rrd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis_tools"))

from visualize_occupancy import CAM_ORDER, build_dataset, build_model, denormalize_for_display, find_occ_gt_indices  # noqa: E402

from .colors import annotation_context
from .core import log_occupancy_grid


def _build_token_index(dataset) -> dict[str, int]:
    return {info["token"]: i for i, info in enumerate(dataset.data_infos)}


def _forward_chain_length(dataset, token_to_idx: dict[str, int], start_idx: int, cap: int) -> int:
    """How many frames are reachable forward from `start_idx` via `next`, capped at `cap`."""
    count = 1
    idx = start_idx
    while count < cap:
        next_token = dataset.data_infos[idx].get("next", "")
        next_idx = token_to_idx.get(next_token) if next_token else None
        if next_idx is None:
            break
        idx = next_idx
        count += 1
    return count


def _resolve_start_index(dataset, token_to_idx: dict[str, int], start_token: str | None, num_frames: int) -> int:
    if start_token is not None:
        if start_token not in token_to_idx:
            raise ValueError(f"token {start_token!r} not found in this dataset split")
        return token_to_idx[start_token]
    # Default: start at a sample with real occupancy GT, so the sequence
    # isn't empty of GT comparisons -- but a GT sample can land anywhere in
    # its scene, including right near the end, which would cut the
    # requested --num-frames short for no good reason. Check a handful of
    # candidates (shuffled, like visualize_occupancy.py's sample picker) and
    # prefer one with enough room left to deliver the full length asked for.
    candidates = find_occ_gt_indices(dataset, num_samples=20, seed=0)
    best_idx, best_len = candidates[0], 0
    for idx in candidates:
        length = _forward_chain_length(dataset, token_to_idx, idx, cap=num_frames)
        if length > best_len:
            best_idx, best_len = idx, length
        if length >= num_frames:
            break
    return best_idx


def _walk_frame_indices(dataset, token_to_idx: dict[str, int], start_idx: int, num_frames: int) -> list[int]:
    """Follow `info['next']` tokens forward from `start_idx`, up to `num_frames` steps.

    Stops early at the end of a scene (`next == ''`) or if the chain leads
    somewhere outside this dataset split (e.g. filtered out), rather than
    raising -- a shorter-than-requested sequence is fine, an error isn't.
    """
    indices = [start_idx]
    idx = start_idx
    while len(indices) < num_frames:
        next_token = dataset.data_infos[idx].get("next", "")
        if not next_token:
            break
        next_idx = token_to_idx.get(next_token)
        if next_idx is None:
            break
        indices.append(next_idx)
        idx = next_idx
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Log a sequence of nuScenes frames (camera images + GT/predicted occupancy) into Rerun.",
    )
    parser.add_argument("config", help="VAD config file (must have occ_head + occ_gt_root configured)")
    parser.add_argument("checkpoint", help="Model checkpoint (.pth) to run predictions from")
    parser.add_argument("--start-token", default=None, help="Sample token to start from (default: first sample with occupancy GT)")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of consecutive samples to log")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--entity-path-prefix", default="", help="Prefix for all logged entity paths")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--save", default=None, help="Save the recording to this .rrd file instead of spawning a viewer")
    output.add_argument("--connect", default=None, help="Connect to an existing Rerun viewer/server instead of spawning one")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prefix = args.entity_path_prefix

    def entity(name: str) -> str:
        return f"{prefix}/{name}" if prefix else name

    from mmengine.config import Config

    cfg = Config.fromfile(args.config)
    print("Building dataset and model once -- this takes a minute...")
    dataset = build_dataset(cfg)
    model = build_model(cfg, args.checkpoint, args.device)

    pc_range = dataset.pc_range
    bev_h, bev_w = dataset.bev_size
    cell_size = ((pc_range[3] - pc_range[0]) / bev_w, (pc_range[4] - pc_range[1]) / bev_h)

    token_to_idx = _build_token_index(dataset)
    start_idx = _resolve_start_index(dataset, token_to_idx, args.start_token, args.num_frames)
    frame_indices = _walk_frame_indices(dataset, token_to_idx, start_idx, args.num_frames)
    print(f"Logging {len(frame_indices)} frame(s) starting from token "
          f"{dataset.data_infos[start_idx]['token']}")

    rr.init("nuscenes_rerun_sequence", spawn=args.save is None and args.connect is None)
    if args.save:
        rr.save(args.save)
    elif args.connect:
        rr.connect_grpc(args.connect)

    rr.log(entity("world"), rr.AnnotationContext(annotation_context()), static=True)

    n_gt, n_pred = 0, 0
    for i, idx in enumerate(frame_indices):
        rr.set_time("frame", sequence=i)
        raw_info = dataset.data_infos[idx]
        token = raw_info["token"]
        sample = dataset[idx]

        img_queue = sample["inputs"]["img"]  # (queue_len, num_cam, C, H, W)
        img_curr = img_queue[-1]
        cam_names = CAM_ORDER[: img_curr.shape[0]]
        for cam_i, cam_name in enumerate(cam_names):
            rr.log(entity(f"camera/{cam_name}"), rr.Image(denormalize_for_display(img_curr[cam_i])))

        class_grid, height_grid = dataset.get_occ_gt(token)
        if class_grid is not None:
            logged = log_occupancy_grid(
                entity("world/occupancy_gt"), class_grid.astype(np.uint8), height_grid, pc_range, cell_size,
            )
            n_gt += int(logged)

        img_batch = img_queue.unsqueeze(0)
        cls_logits, height_pred = model.predict_occ(img_batch, [sample["data_samples"]])
        pred_class = cls_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred_height = height_pred[0].cpu().numpy()
        log_occupancy_grid(entity("world/occupancy_pred"), pred_class, pred_height, pc_range, cell_size)
        n_pred += 1

        print(f"  [{i + 1}/{len(frame_indices)}] token={token} "
              f"({'GT present' if class_grid is not None else 'no GT'})")

    print(f"Done: {n_pred} frame(s) logged, {n_gt} with real occupancy GT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
