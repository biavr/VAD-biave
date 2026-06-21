#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import os.path as osp
import types
import argparse
from json import dump
import tempfile
import numpy as np

try:
    import mmdet3d.evaluation.metrics.nuscenes_metric as nm
    from mmengine.fileio import load as mmengine_load

    # Secure the pristine underlying library calculation reference
    orig_compute_metrics = nm.NuScenesMetric.compute_metrics

    def patched_compute_metrics(metric_instance, results):
        if not hasattr(metric_instance, 'dataset_meta') or metric_instance.dataset_meta is None:
            metric_instance.dataset_meta = dict()
        guessed_version = 'v1.0-trainval' if len(results) > 500 else 'v1.0-mini'
        if 'version' not in metric_instance.dataset_meta:
            metric_instance.dataset_meta['version'] = guessed_version
        elif metric_instance.dataset_meta['version'] == 'v1.0-mini' and len(results) > 500:
            metric_instance.dataset_meta['version'] = 'v1.0-trainval'
        return orig_compute_metrics(metric_instance, results)

    nm.NuScenesMetric.compute_metrics = patched_compute_metrics
    print("======== [VAD PATCH] Successfully injected ultimate token alignment interceptor! ========")
except Exception as e:
    print(f"======== [VAD PATCH WARNING] Failed to inject metrics patch: {str(e)} ========")

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
    parser.add_argument('checkpoint', nargs='?', default=None, help='checkpoint file path')
    parser.add_argument('--work-dir', help='the directory to save evaluation metrics')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval', type=str)
    parser.add_argument('--tmpdir', type=str)
    parser.add_argument('--predictions', default=None, help='Path to results_nusc.json file')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher

    # ======================================================================
    # ⚡ DEFINITIVE NATIVE OFFLINE EVALUATION BYPASS 
    # ======================================================================
    if hasattr(args, 'predictions') and args.predictions is not None:
        print(f"\n[VAD OFFLINE MODE] Loading pre-computed predictions from: {args.predictions}")
        import mmengine
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        from nuscenes.eval.common.config import config_factory

        # 1. Load your pre-computed data dictionary payload
        raw_data = mmengine.load(args.predictions)
        
        # 2. Extract internal results dictionary map safely
        results_payload = raw_data
        if isinstance(results_payload, dict) and 'results' in results_payload:
            results_payload = results_payload['results']
        if isinstance(results_payload, dict) and 'results' in results_payload:
            results_payload = results_payload['results']

        print(f"[VAD OFFLINE MODE] Target data payload resolved. Frames to evaluate: {len(results_payload)}")

        print("[VAD OFFLINE MODE] Initializing native NuScenes database tracking anchors...")
        nusc_root = '/workspace/datasets/nuscenes/v1.0-trainval'
        nusc_version = 'v1.0-trainval'
        
        # Instantiate pristine native tracker
        nusc_db = NuScenes(version=nusc_version, dataroot=nusc_root, verbose=False)
        eval_config = config_factory('detection_cvpr_2019')

        # ======================================================================
        # DYNAMIC GLOBAL COORDINATE TRANSFORMATION LAYER
        # ======================================================================
        print("[VAD OFFLINE MODE] Transforming local predictions into Global Map coordinates...")
        from pyquaternion import Quaternion
        
        global_results = {}
        for token, box_list in results_payload.items():
            # Fetch the calibration matrices for this specific scene token frame
            sample_record = nusc_db.get('sample', token)
            sd_record = nusc_db.get('sample_data', sample_record['data']['LIDAR_TOP'])
            
            # Fetch translation and rotation from sensor to ego-vehicle
            calib_sensor = nusc_db.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            # Fetch translation and rotation from ego-vehicle to global map
            pose_ego = nusc_db.get('ego_pose', sd_record['ego_pose_token'])
            
            transformed_boxes = []
            for box in box_list:
                # Initialize local coordinate arrays
                pos = np.array(box['translation'])
                rot = Quaternion(box['rotation'])
                
                # A. Local Sensor Frame -> Ego Vehicle Frame
                sensor_rot_q = Quaternion(calib_sensor['rotation'])
                pos = np.dot(sensor_rot_q.rotation_matrix, pos) + np.array(calib_sensor['translation'])
                rot = sensor_rot_q * rot
                
                # B. Ego Vehicle Frame -> Global Earth Frame
                ego_rot_q = Quaternion(pose_ego['rotation'])
                pos = np.dot(ego_rot_q.rotation_matrix, pos) + np.array(pose_ego['translation'])
                rot = ego_rot_q * rot
                
                # Construct the matching dictionary layout safely
                transformed_box = box.copy()
                transformed_box['translation'] = pos.tolist()
                transformed_box['rotation'] = rot.elements.tolist()
                transformed_boxes.append(transformed_box)
                
            global_results[token] = transformed_boxes
        # ======================================================================

        # 3. Create the submission metadata structure the devkit parser tracks
        submission_dict = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external_tracking": False
            },
            "results": global_results
        }

        # 4. Run the file-based devkit evaluator inside a temporary folder context
        permanent_eval_dir = '/workspace/logs/permanent_results/metrics_plots'
        os.makedirs(permanent_eval_dir, exist_ok=True)
        
        res_mock_path = osp.join(permanent_eval_dir, 'submission_output.json')
        with open(res_mock_path, 'w') as f:
            dump(submission_dict, f)

        print("[VAD OFFLINE MODE] Starting official NuScenesEval execution engine...")
        evaluator_engine = NuScenesEval(
            nusc=nusc_db,
            config=eval_config,
            result_path=res_mock_path,
            eval_set='val',
            output_dir=permanent_eval_dir,  # 🌟 Directs plots and JSONs to stay forever!
            verbose=True
        )

        print("\n================ OFFICIAL DEVKIT RUNNING ================")
        eval_output = evaluator_engine.main(plot_examples=False)
        metrics_summary = eval_output[0] if isinstance(eval_output, tuple) else eval_output
        print("=========================================================\n")
        
        print("================ EVALUATION SUMMARY METRICS ================")
        import pprint
        pprint.pprint(metrics_summary)
        print("============================================================")
                
        return  # Terminate early and safely!
    # ======================================================================

    if args.checkpoint is None:
        raise ValueError("The 'checkpoint' argument position is required unless running in offline mode via --predictions")

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()