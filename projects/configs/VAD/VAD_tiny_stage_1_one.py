_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3 # each sequence contains `queue_length` frames.
total_epochs = 48

# 1. Add Custom Imports (Crucial for VAD plugin)
custom_imports = dict(
    imports=['projects.mmdet3d_plugin'], 
    allow_failed_imports=False
)

# ... [Keep your variables like point_cloud_range, _dim_, etc. unchanged] ...

# 2. Model Refactoring
model = dict(
    type='VAD',
    # Add Data Preprocessor (Modern replacement for Normalize/Pad in pipeline)
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    use_grid_mask=True,
    video_test_mode=True,
    # Use init_cfg for pretrained weights
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), 
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    # ... pts_bbox_head remains largely the same structure ...
)

# 3. Data Loader Refactoring (Modern Style)
train_dataloader = dict(
    batch_size=1, # Replaces samples_per_gpu
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        # ... rest of your dataset keys ...
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type)
)
test_dataloader = val_dataloader

# 4. Optimization Refactoring
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2) # Moved from optimizer_config
)

# 5. Training/Validation Loops (Replaces runner and evaluation)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 6. Param Scheduler (Replaces lr_config)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=total_epochs, begin=0, end=total_epochs, by_epoch=True, eta_min_ratio=1e-3)
]

# 7. Evaluators
val_evaluator = dict(
    type='NuScenesMetric', 
    data_root=data_root,
    ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator