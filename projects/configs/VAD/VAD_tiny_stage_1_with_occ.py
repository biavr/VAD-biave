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

# Iteration-based (not epoch-based) checkpoints: the live dashboard's
# occupancy watcher (tools/live_dashboard/occ_watcher.py) re-renders
# predictions against whatever is newest, so this is what sets how often
# that preview updates. max_keep_ckpts avoids filling the disk with one
# checkpoint every 500 steps over a long run.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, max_keep_ckpts=3)
)

# Longer run than the base config's total_epochs=3: the occ head's star
# artifact (see tools/live_dashboard -- camera-FOV-boundary bias in
# BEVFormer's spatial cross-attention) was still visibly fading at epoch 3,
# and occupancy GT only covers ~6% of samples, so it needs more passes over
# the data than the other heads to converge. train_cfg/param_scheduler are
# overridden directly (not just total_epochs) since the base config's
# param_scheduler milestones are plain numbers computed from total_epochs at
# load time, not a live reference that an override here would reach.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[7, 13],
        gamma=0.1)
]

work_dir = '/workspace/logs/outputs_tiny_stage_1_with_occ'
