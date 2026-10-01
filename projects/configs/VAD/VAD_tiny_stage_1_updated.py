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
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_classes = len(map_classes)
map_fixed_ptsnum_per_line = 20  # must match map_num_pts_per_gt_vec below

# --- tiny: lighter backbone, single BEV feature level, smaller BEV grid,
# fewer queries and fewer transformer layers than VAD_base_stage_1_updated.py ---
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3
total_epochs = 15

data_root = '/workspace/datasets/nuscenes/v1.0-trainval/'

model = dict(
    type='VAD',
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        embed_dims=_dim_,
        bev_h=bev_h_,
        bev_w=bev_w_,

        map_num_vec=100,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_line,
        map_code_size=2,
        valid_fut_ts=6,
        traj_num_cls=6,
        fut_mode=6,
        fut_ts=6,

        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_pe=True,
        score_thresh=0.4,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',

        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=100,
            map_num_pts_per_vec=map_fixed_ptsnum_per_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            num_feature_levels=_num_levels_,
            num_cams=6,
            two_stage_num_proposals=300,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),

        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1)],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1)],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1)],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1)],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),

        bbox_coder=dict(
            type='projects.mmdet3d_plugin.core.bbox.coders.fut_nms_free_coder.CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='projects.mmdet3d_plugin.core.bbox.coders.map_nms_free_coder.MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),

        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),

        # Stage 1 = Perception & Prediction (detection + map + motion trained,
        # planning muted). Flip these two back to 0.0 if you want detection-only.
        loss_traj=dict(type='mmdet.L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
        loss_map_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_map_bbox=dict(type='mmdet.L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),

        # Planning stays muted -- this is stage 2's job.
        loss_plan_reg=dict(type='mmdet.L1Loss', loss_weight=0.0),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=0.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.0),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            assigner=dict(
                type='mmdet.HungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
                pc_range=point_cloud_range),
            map_assigner=dict(
                type='mmdet.MapHungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='mmdet.BBox3DL1Cost', weight=0.0),
                iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='mmdet.OrderedPtsL1Cost', weight=1.0),
                pc_range=point_cloud_range))),
)

train_pipeline = [
    dict(type='mmdet3d.LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='projects.mmdet3d_plugin.datasets.pipelines.PhotoMetricDistortionMultiViewImage'),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='projects.mmdet3d_plugin.datasets.pipelines.CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='projects.mmdet3d_plugin.datasets.pipelines.CustomObjectNameFilter', classes=class_names),
    dict(type='projects.mmdet3d_plugin.datasets.pipelines.RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='mmdet3d.Pack3DDetInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=[
             'lidar2img', 'can_bus', 'timestamp', 'sample_idx', 'img_metas',
             'gt_attr_labels',
             'gt_ego_his_trajs', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
             'gt_ego_fut_cmd', 'gt_ego_lcf_feat',
             'box_mode_3d', 'box_type_3d', 'point_cloud_range'
         ]),
]

test_pipeline = [
    dict(type='mmdet3d.LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='mmdet3d.LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
]

train_dataloader = dict(
    # NOTE: batch_size>1 reliably produces NaN predictions within the first
    # ~250 iterations (confirmed independent of AMP and of LR -- it crashed
    # identically at lr=8e-4 w/ AMP, lr=2e-4 w/ AMP, and lr=2e-4 in plain
    # fp32). VADCustomNuScenesDataset's per-sample map/instance handling was
    # evidently only ever built and tested at batch_size=1; something in that
    # path (likely GT map-vector padding/collation across a batch) breaks
    # when batch_size>1. To get more throughput, use more GPUs instead (each
    # still running batch_size=1) rather than increasing this value.
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmengine.DefaultSampler', shuffle=True),
    dataset=dict(
        type='VADCustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        filter_empty_gt=False,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
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
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_line))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet3d.NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    # NOTE: AmpOptimWrapper (fp16) caused grad_norm to go NaN at iter ~250-600
    # across two separate runs, at different LRs and before/after warmup --
    # the common factor was AMP itself, likely fp16 overflow/underflow in one
    # of the custom ops (deformable attention or a map/chamfer loss). Back to
    # plain fp32 until that's root-caused; batch_size=4 alone is unaffected.
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        # NOTE: do not scale this with batch_size. 8e-4 (linear-scaled for
        # batch_size 1->4) caused grad_norm to go NaN right after warmup
        # ended (iter ~500), which crashed the Hungarian map assigner a
        # hundred-odd iterations later. 2e-4 is the proven-stable value.
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

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
        end=total_epochs,
        by_epoch=True,
        milestones=[total_epochs/2, total_epochs-2],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
# Validation is disabled: val_evaluator below is mmdet3d's stock NuScenesMetric,
# which expects dataset_meta['version'] and the stock nuScenes annotation format.
# VADCustomNuScenesDataset's real eval logic (detection + map + motion) still
# lives in its old-style .evaluate()/_format_bbox(), which was never ported to
# the new BaseMetric API this mmengine version requires, so the stock metric
# crashes with `KeyError: 'version'` at the end of epoch 1. Re-enable once a
# proper BaseMetric subclass exists for VAD's custom outputs.
# mmengine requires val_dataloader/val_cfg/val_evaluator to be all-None or
# all-set, so null out the other two as well (test_dataloader keeps its own
# dict -- it was assigned val_dataloader's value by copy, above, before this).
val_dataloader = None
val_evaluator = None
val_cfg = None
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Needed for DDP once you're back on multi-GPU -- VADHead's branches don't
# necessarily all contribute to every forward pass's graph.
find_unused_parameters = True

custom_hooks = [dict(type='CustomSetEpochInfoHook')]

work_dir = '/workspace/logs/outputs_tiny_stage_1'
