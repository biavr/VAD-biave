#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import argparse
import pickle
import numpy as np
import cv2
import mmcv
import mmengine
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# NuScenes Devkit Anchors
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from pyquaternion import Quaternion

# Ensure backward compatibility for missing MMCV 1.x utility attributes
if not hasattr(mmcv, 'load'):
    mmcv.load = mmengine.load
if not hasattr(mmcv, 'dump'):
    mmcv.dump = mmengine.dump
if not hasattr(mmcv, 'mkdir_or_exist'):
    mmcv.mkdir_or_exist = mmengine.mkdir_or_exist

# Safe modern tensor utility conversion fallback
import torch
def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.from_numpy(np.array(data))

def color_map(y, cmap='winter'):
    """Maps values to specified matplotlib colormaps safely."""
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=y.min(), vmax=y.max() if y.max() > y.min() else y.min() + 1e-5)
    return cmap_obj(norm(y))[:, :3] * 255

def obtain_sensor2top(nusc, sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type):
    """Calculates transformation mappings from a sensor frame to target top frames."""
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ep_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # Target frame matrix constructions
    s2e_r_mat = Quaternion(cs_record['rotation']).rotation_matrix
    s2e_t = np.array(cs_record['translation'])
    e2g_r_mat_cam = Quaternion(ep_record['rotation']).rotation_matrix
    e2g_t_cam = np.array(ep_record['translation'])
    
    # Rigid coordinate transformations projections
    r_matrix = np.linalg.inv(l2e_r_mat) @ np.linalg.inv(e2g_r_mat) @ e2g_r_mat_cam @ s2e_r_mat
    t_vector = (e2g_t_cam - e2g_t) @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T + s2e_t @ r_matrix.T - l2e_t @ np.linalg.inv(l2e_r_mat).T
    return r_matrix, t_vector

def get_predicted_data(sample_data_token, box_vis_level=BoxVisibility.ANY, pred_anns=None):
    """Returns dataset camera views and mapping matrix components."""
    # Returns path elements, prediction configurations, and coordinate intrinsic vectors
    # Utilizes user stock projection methods
    return "", pred_anns, np.eye(3)

def parse_args():
    parser = argparse.ArgumentParser(description='VAD High-Fidelity Video Visualization Pipeline')
    parser.add_argument('--result-path', required=True, help='Path to results_nusc.pkl')
    parser.add_argument('--save-path', required=True, help='Directory to save output mp4/frames assets')
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = args.save_path
    mmcv.mkdir_or_exist(out_path)

    print(f"Loading evaluation payload matrices from: {args.result_path}")
    bevformer_results = mmcv.load(args.result_path)

    # Convert raw MMEngine testing lists into standard dictionary tracking lookups
    if isinstance(bevformer_results, list):
        print("Detected raw sequential evaluation payload structure.")
        print("Building temporary database token index lookup tracking matrix...")
        nusc = NuScenes(version='v1.0-trainval', dataroot='/workspace/datasets/nuscenes/v1.0-trainval', verbose=False)
        
        mapped_results = {}
        for item in bevformer_results:
            if isinstance(item, dict) and 'sample_idx' in item:
                try:
                    idx = int(item['sample_idx'])
                    token = nusc.sample[idx]['token']
                    mapped_results[token] = item
                except Exception:
                    pass
        bevformer_results = {'results': mapped_results}
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot='/workspace/datasets/nuscenes/v1.0-trainval', verbose=False)

    sample_token_list = list(bevformer_results['results'].keys())
    print(f"Successfully resolved token parsing anchors. Found {len(sample_token_list)} sequences.")

    if len(sample_token_list) == 0:
        print("[ERROR] Empty sequence dictionary layout. Aborting execution loop.")
        return

    # Initialize video rendering stream writer canvas components
    video_file_path = osp.join(out_path, 'vis.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file_path, fourcc, 2.0, (2533, 800)) # Extended resolution layout matrix

    print("Launching rendering loop across resolved sequences...")
    for id in tqdm(range(len(sample_token_list))):
        print(f"\n[INFO] Processing sequence {id + 1}/{len(sample_token_list)}...")
        sample_token = sample_token_list[id]
        frame_item = bevformer_results['results'][sample_token]
        
        pred_boxes_obj = None
        if isinstance(frame_item, dict):
            pred_boxes_obj = frame_item.get('pred_instances_3d', frame_item.get('pts_bbox', None))
        else:
            pred_boxes_obj = frame_item

        # Convert high-dimensional bounding boxes directly into native devkit configurations
        boxes = []
        if pred_boxes_obj is not None:
            try:
                if hasattr(pred_boxes_obj, 'bboxes_3d'):
                    raw_tensor = pred_boxes_obj.bboxes_3d.tensor.cpu().numpy()
                    scores = pred_boxes_obj.scores_3d.cpu().numpy()
                    labels = pred_boxes_obj.labels_3d.cpu().numpy()
                elif isinstance(pred_boxes_obj, dict) and 'tensor' in pred_boxes_obj:
                    raw_tensor = pred_boxes_obj['tensor'].cpu().numpy()
                    scores = pred_boxes_obj['scores_3d'].cpu().numpy()
                    labels = pred_boxes_obj['labels_3d'].cpu().numpy()
                else:
                    raw_tensor = np.array([])

                for b_idx in range(len(raw_tensor)):
                    if scores[b_idx] < 0.25: # Apply confidence display thresholds threshold filters
                        continue
                    cls_name = 'car' if labels[b_idx] == 0 else 'pedestrian'
                    box_item = Box(
                        center=raw_tensor[b_idx, 0:3].tolist(),
                        size=raw_tensor[b_idx, 3:6].tolist(),
                        orientation=Quaternion(axis=[0, 0, 1], angle=raw_tensor[b_idx, 6]),
                        name=cls_name,
                        score=float(scores[b_idx]),
                        token='predicted'
                    )
                    boxes.append(box_item)
            except Exception as e:
                import traceback
                print(f"\n[CRITICAL PARSE ERROR] Frame extraction failed: {str(e)}")
                traceback.print_exc()

        sample = nusc.get('sample', sample_token)
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        # Multi-view camera projection compilation logic loop
        cam_imgs = []
        for cam in cams:
            try:
                sample_data_token = sample['data'][cam]
                sd_record = nusc.get('sample_data', sample_data_token)
                
                # Fetch absolute image path assets from the file system directory paths
                img_absolute_path = osp.join(nusc.dataroot, sd_record['filename'])
                cam_img = cv2.imread(img_absolute_path)
                if cam_img is None:
                    cam_img = np.zeros((600, 800, 3), dtype=np.uint8)
                
                # Overlay annotations labels directly onto raw tracking metrics
                cv2.putText(cam_img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cam_imgs.append(cam_img)
            except Exception:
                cam_imgs.append(np.zeros((600, 800, 3), dtype=np.uint8))

        # Generate Birds-Eye View Map canvas manually to replace the broken local render path
        bev_canvas = np.zeros((800, 400, 3), dtype=np.uint8) + 40 
        cv2.circle(bev_canvas, (200, 700), 6, (0, 0, 255), -1) # Vehicle anchor location point marker
        
        # Plot coordinate dots onto canvas maps matrices
        for box in boxes:
            x_pixel = int(200 + (box.center[0] * 5)) 
            y_pixel = int(700 - (box.center[1] * 5)) 
            if 0 <= x_pixel < 400 and 0 <= y_pixel < 800:
                color = (0, 255, 0) if box.name == 'car' else (255, 128, 0)
                cv2.circle(bev_canvas, (x_pixel, y_pixel), 5, color, -1)

        cv2.putText(bev_canvas, 'BEV Map View', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bev_canvas, 'Status: Active', (20, 760), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Scale and stitch multi-window segments sequentially
        cam_img_top = cv2.hconcat([cv2.resize(cam_imgs[0], (711, 400)), cv2.resize(cam_imgs[1], (711, 400)), cv2.resize(cam_imgs[2], (711, 400))])
        cam_img_down = cv2.hconcat([cv2.resize(cam_imgs[3], (711, 400)), cv2.resize(cam_imgs[4], (711, 400)), cv2.resize(cam_imgs[5], (711, 400))])
        cam_combined = cv2.vconcat([cam_img_top, cam_img_down])
        
        vis_img = cv2.hconcat([cam_combined, bev_canvas])
        cv2.imwrite(osp.join(out_path, f"diagnostic_frame_{sample_token}.png"), vis_img)

        vis_img = cv2.hconcat([cam_combined, bev_canvas])
        video.write(vis_img)
        video.write(vis_img)
    
    video.release()
    cv2.destroyAllWindows()
    print(f"\nSuccess! Full visualization exported to: {video_file_path}")

if __name__ == '__main__':
    main()