"""VAD_tiny_stage_1 + an auxiliary 2.5D BEV occupancy head.

Everything (backbone, BEV encoder, detection/map/motion heads, dataset,
optimizer, schedule) is inherited unchanged from VAD_tiny_stage_1_updated.py.
This file only adds:
  - model.occ_head: trained jointly with the rest, on the same BEV feature
    map pts_bbox_head already computes (see occupancy_head.py).
  - train_dataloader.dataset.occ_gt_root: points the dataset at the GT
    produced by tools/data_converter/occupancy_gt_generator.py so it can
    attach a (class_grid, height_grid) target to each sample.
"""
_base_ = ['./VAD_tiny_stage_1_updated.py']

model = dict(
    occ_head=dict(
        type='VADOccHead',
        in_channels=256,  # must match _dim_ in VAD_tiny_stage_1_updated.py
        feat_channels=256,
        num_classes=32,   # official nuScenes-lidarseg class count
    )
)

train_dataloader = dict(
    dataset=dict(
        occ_gt_root='/workspace/datasets/nuscenes/v1.0-trainval/occupancy_gt',
    )
)

work_dir = '/workspace/logs/outputs_tiny_stage_1_with_occ'
