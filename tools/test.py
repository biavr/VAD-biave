#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import os.path as osp
import types
import argparse
import pickle
import numpy as np

# ======================================================================
# MMEngine & MMDet3D 3.x FRAMEWORK COMPATIBILITY LAYER
# ======================================================================
GLOBAL_OUT_PATH = None  # Safe container for runtime pickling intercept

try:
    import mmdet3d.evaluation.metrics.nuscenes_metric as nm
    from mmengine.fileio import load as mmengine_load

    # Secure the pristine underlying library calculation reference
    orig_compute_metrics = nm.NuScenesMetric.compute_metrics

    def patched_compute_metrics(metric_instance, results):
        """Intercepts the metrics builder engine to resolve runtime metadata mismatches."""
        if not hasattr(metric_instance, 'dataset_meta') or metric_instance.dataset_meta is None:
            metric_instance.dataset_meta = dict()
            
        guessed_version = 'v1.0-trainval' if len(results) > 500 else 'v1.0-mini'
        
        if 'version' not in metric_instance.dataset_meta:
            metric_instance.dataset_meta['version'] = guessed_version
        elif metric_instance.dataset_meta['version'] == 'v1.0-mini' and len(results) > 500:
            metric_instance.dataset_meta['version'] = 'v1.0-trainval'
            
        if 'classes' not in metric_instance.dataset_meta:
            metric_instance.dataset_meta['classes'] = [
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
            ]

        # 🚀 RE-ROUTE LEGACY ANNOTATION STRUCTS TO PREVENT THE KEYERROR
        if hasattr(metric_instance, 'ann_file') and metric_instance.ann_file:
            try:
                raw_pkl = mmengine_load(metric_instance.ann_file)
                if isinstance(raw_pkl, dict) and 'data_list' not in raw_pkl:
                    # Find alternative data storage names used in custom/legacy files
                    legacy_data = raw_pkl.get('infos', raw_pkl.get('data', list(raw_pkl.values())[0]))

                    # The legacy VAD info dicts carry pose as flat
                    # lidar2ego_rotation/translation + ego2global_rotation/translation
                    # (quaternion + vector). The stock NuScenesMetric's
                    # lidar_nusc_box_to_global() expects the new-style mmdet3d 3.x
                    # shape instead -- info['lidar_points']['lidar2ego'] and
                    # info['ego2global'] as precomputed 4x4 matrices -- and crashes
                    # with a KeyError on 'lidar_points' otherwise. Add those fields
                    # (same construction nuscenes_vad_dataset.py already uses
                    # elsewhere) without touching the legacy fields anything else
                    # still reads.
                    from pyquaternion import Quaternion
                    for info in legacy_data:
                        if 'lidar_points' not in info or 'lidar2ego' not in info.get('lidar_points', {}):
                            lidar2ego = np.eye(4)
                            lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                            lidar2ego[:3, 3] = info['lidar2ego_translation']
                            info.setdefault('lidar_points', {})['lidar2ego'] = lidar2ego.tolist()
                        if 'ego2global' not in info:
                            ego2global = np.eye(4)
                            ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
                            ego2global[:3, 3] = info['ego2global_translation']
                            info['ego2global'] = ego2global.tolist()

                    raw_pkl['data_list'] = legacy_data

                    # Override the loader definition context inside the specific module namespace
                    import mmdet3d.evaluation.metrics.nuscenes_metric as native_metric_mod
                    native_metric_mod.load = lambda *args, **kwargs: raw_pkl

                    # The "match tokens" block right below needs
                    # metric_instance.data_infos, but that attribute is
                    # normally only set *inside* the stock compute_metrics()
                    # we're about to call (via the load() override above,
                    # consumed deeper in format_results()) -- i.e. too late
                    # for the block below to see it on this call. Set it
                    # directly here so that block actually has something to
                    # match against, instead of silently no-op'ing (which is
                    # exactly what was happening -- every result's sample_idx
                    # stayed whatever VAD.py's predict() put there, which due
                    # to a separate bug was always 0).
                    metric_instance.data_infos = legacy_data
            except Exception:
                pass

        # Match incoming result items with the true tracking annotation tokens.
        # Relies on `results` being in the same order as the (unshuffled,
        # single-process) test dataloader -- true for how this is actually
        # run here, but would need rethinking under distributed/shuffled eval.
        if hasattr(metric_instance, 'data_infos') and metric_instance.data_infos and results:
            for idx, res in enumerate(results):
                if isinstance(res, dict):
                    safe_idx = idx % len(metric_instance.data_infos)
                    true_token = metric_instance.data_infos[safe_idx].get('token', None)
                    if true_token:
                        res['sample_idx'] = int(safe_idx)
                        res['sample_token'] = str(true_token)
                        if 'pred_instances_3d' in res and isinstance(res['pred_instances_3d'], dict):
                            res['pred_instances_3d']['sample_token'] = str(true_token)
                        if 'pts_bbox' in res and isinstance(res['pts_bbox'], dict):
                            res['pts_bbox']['sample_token'] = str(true_token)
                            res['pts_bbox']['sample_idx'] = int(safe_idx)

        # Intercept results and dump to pkl before running devkit
        global GLOBAL_OUT_PATH
        if GLOBAL_OUT_PATH is not None:
            print(f"\n[VAD COMPATIBILITY] Intercepted test results payload! Saving binary PKL file to: {GLOBAL_OUT_PATH}")
            os.makedirs(osp.dirname(osp.abspath(GLOBAL_OUT_PATH)), exist_ok=True)
            with open(GLOBAL_OUT_PATH, 'wb') as f:
                pickle.dump(results, f)
            print("[VAD COMPATIBILITY] Binary PKL saved successfully.")

        return orig_compute_metrics(metric_instance, results)

    # Inject the runtime interceptor globally
    nm.NuScenesMetric.compute_metrics = patched_compute_metrics
    print("======== [VAD COMPATIBILITY] MMEngine metrics patches injected successfully! ========")
except Exception as e:
    print(f"======== [VAD WARNING] Metrics compatibility interceptor bypassed: {str(e)} ========")

# MMCV Utils Backward Compatibility Shims
from mmengine.utils.version_utils import digit_version
from mmengine.utils.dl_utils.parrots_wrapper import TORCH_VERSION

try:
    import mmcv.utils
except ImportError:
    import mmcv
    mmcv.utils = types.ModuleType('mmcv.utils')
    sys.modules['mmcv.utils'] = mmcv.utils

mmcv.utils.TORCH_VERSION = TORCH_VERSION
mmcv.utils.digit_version = digit_version
# ======================================================================

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Automatic Path Resolution
current_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.abspath(osp.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='MMEngine Standard Test Script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the directory to save evaluation metrics')
    parser.add_argument('--out', default=None, help='output result file destination in pickle format')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval', type=str, help='evaluation metrics')
    parser.add_argument('--tmpdir', type=str, help='temporary directory directory storage')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    global GLOBAL_OUT_PATH
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Setup execution paths
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher

    # Set the global output path tracker safely without mutating the evaluator constructor configs
    if args.out is not None:
        GLOBAL_OUT_PATH = args.out

    # Initialize MMEngine environment loop runner
    runner = Runner.from_cfg(cfg)
    
    # Run comprehensive model inference
    runner.test()

if __name__ == '__main__':
    main()