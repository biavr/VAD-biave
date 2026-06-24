# ==============================================================================
# VAD Base Stage-1 MMEngine Modern Configuration
# Optimized for high-fidelity perception backbone training on nuScenes full trainval
# ==============================================================================

# Inherit baseline environment components safely from modern configurations
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
default_scope = 'mmdet3d'

custom_imports = dict(
    imports=['mmdet.models.layers.transformer',
             'mmdet.models.layers',
             'mmdet3d', 
             'mmdet3d.models.data_preprocessors',
             'mmdet.models.losses',
             'projects.mmdet3d_plugin',
             'projects.mmdet3d_plugin.core.bbox.coders.fut_nms_free_coder',
             'projects.mmdet3d_plugin.core.bbox.coders.map_nms_free_coder',
             'projects.mmdet3d_plugin.datasets.pipelines', 
             ], 
    allow_failed_imports=False
)

point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_classes = len(map_classes)
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.
total_epochs = 48

data_root = '/workspace/datasets/nuscenes/v1.0-trainval/'

model = dict(
    type='VAD', # Central model driver pointing to projects/mmdet3d_plugin/
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='open-mmlab://resnet101'),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='VADTransformer',
            rotate_step=1,
            num_feature_levels=4,
            num_cams=6,
            two_stage_num_proposals=900,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_collection=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='VADTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomProtoAttention', # Re-routes custom VAD structures
                            embed_dims=256,
                            num_levels=4,
                            num_points=4)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10),
        map_bbox_coder=dict(
            type='projects.mmdet3d_plugin.core.bbox.coders.map_nms_free_coder.MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=50,
            col_num_embed=50),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIOULoss', loss_weight=0.0),
        
        # Planning losses are muted or decoupled to anchor backbone training cleanly
        loss_map_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_map_reg=dict(type='mmdet.L1Loss', loss_weight=1.0),
        loss_traj_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
        loss_traj_reg=dict(type='mmdet.L1Loss', loss_weight=0.0)),
    
    # Model Model Config Training Parameters Tuning Mapping
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            out_size_factor=8,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='mmdet.IoUReviewCost', weight=0.0)))))

train_pipeline = [
    dict(type='mmdet3d.LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='mmdet3d.LoadPointsFromMultiViewImages', replace_img_with_black=False),
    dict(type='mmdet3d.PhotoMetricDistortionMultiViewImage'),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_map_3d=True, with_traj_3d=True),
    dict(type='mmdet3d.ObjectRangeFilter3D', point_cloud_range=point_cloud_range),
    dict(type='mmdet3d.ObjectNameFilter3D', classes=[
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]),
    dict(type='mmdet3d.DefaultFormatBundle3D', class_names=[
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]),
    dict(type='mmengine.Pack3DDetInputs', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_maps_3d', 'gt_trajs_3d'])
]

val_pipeline = [
    dict(type='mmdet3d.LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='mmdet3d.LoadPointsFromMultiViewImages', replace_img_with_black=False),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='mmengine.Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]


train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='mmengine.DefaultSampler', shuffle=True),
    dataset=dict(
        type='VADCustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        # --- NEW: Wrap classes in metainfo ---
        metainfo=dict(classes=class_names), 
        filter_empty_gt=False, # Ensure this is False for now
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        # Keep these here, but we will "pop" them in the Python code
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        serialize_data=False,)
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='mmengine.DefaultSampler', shuffle=False),
    dataset=dict(
        type='VADCustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
        pipeline=val_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range))

test_dataloader = val_dataloader


# Evaluation Hooks definition mapping blocks
val_evaluator = dict(
    type='mmdet3d.NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# ------------------------------------------------------------------------------
# 4. Re-engineered Optimization & Scheduling Configurations
# ------------------------------------------------------------------------------
# Encapsulated cleanly inside an optim_wrapper structural layout matrix
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4, # Baseline base stage 1 perception tracking parameter sets
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# Multi-step parameterized decay scheduler tracking structures
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
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# Modern training state controllers loops configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------------------------------------------------------------------
# 5. Core Operational Runtimes & Checkpoint Handlers
# ------------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Ensure visualization artifacts and temporary validation checkpoints remain local
work_dir = '/workspace/logs/outputs_stage_1'