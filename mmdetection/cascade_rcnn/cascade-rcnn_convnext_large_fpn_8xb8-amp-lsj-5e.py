auto_scale_lr = dict(base_batch_size=64)
backend_args = None
batch_augments = [
    dict(size=(
        800,
        800,
    ), type='BatchFixedSizePad'),
]
checkpoint_file = '/data/ephemeral/mmdetection/pretrained/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
data_root = '/data/ephemeral/dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluator = None
image_size = (
    800,
    800,
)
img_scales = [
    (
        1333,
        800,
    ),
    (
        666,
        400,
    ),
]
launcher = 'none'
load_from = './work_dirs/cascade-rcnn_convnext_large_fpn_8xb8-amp-lsj-5e_last/epoch_7.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 5
metainfo = dict(
    classes=(
        'General trash',
        'Paper',
        'Paper pack',
        'Metal',
        'Glass',
        'Plastic',
        'Styrofoam',
        'Plastic bag',
        'Battery',
        'Clothing',
    ))
model = dict(
    module=dict(
        backbone=dict(
            arch='large',
            drop_path_rate=0.5,
            gap_before_final_norm=False,
            init_cfg=dict(
                checkpoint=
                '/data/ephemeral/mmdetection/pretrained/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth',
                prefix='backbone.',
                type='Pretrained'),
            layer_scale_init_value=1.0,
            out_indices=[
                0,
                1,
                2,
                3,
            ],
            type='mmpretrain.ConvNeXt'),
        data_preprocessor=dict(
            batch_augments=[
                dict(size=(
                    800,
                    800,
                ), type='BatchFixedSizePad'),
            ],
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_size_divisor=32,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='DetDataPreprocessor'),
        neck=dict(
            in_channels=[
                192,
                384,
                768,
                1536,
            ],
            num_outs=5,
            out_channels=384,
            type='FPN'),
        roi_head=dict(
            bbox_head=[
                dict(
                    bbox_coder=dict(
                        target_means=[
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        target_stds=[
                            0.1,
                            0.1,
                            0.2,
                            0.2,
                        ],
                        type='DeltaXYWHBBoxCoder'),
                    fc_out_channels=1024,
                    in_channels=384,
                    loss_bbox=dict(
                        beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                    loss_cls=dict(
                        loss_weight=1.0,
                        type='CrossEntropyLoss',
                        use_sigmoid=False),
                    num_classes=10,
                    reg_class_agnostic=True,
                    roi_feat_size=7,
                    type='Shared2FCBBoxHead'),
                dict(
                    bbox_coder=dict(
                        target_means=[
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        target_stds=[
                            0.05,
                            0.05,
                            0.1,
                            0.1,
                        ],
                        type='DeltaXYWHBBoxCoder'),
                    fc_out_channels=1024,
                    in_channels=384,
                    loss_bbox=dict(
                        beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                    loss_cls=dict(
                        loss_weight=1.0,
                        type='CrossEntropyLoss',
                        use_sigmoid=False),
                    num_classes=10,
                    reg_class_agnostic=True,
                    roi_feat_size=7,
                    type='Shared2FCBBoxHead'),
                dict(
                    bbox_coder=dict(
                        target_means=[
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        target_stds=[
                            0.033,
                            0.033,
                            0.067,
                            0.067,
                        ],
                        type='DeltaXYWHBBoxCoder'),
                    fc_out_channels=1024,
                    in_channels=384,
                    loss_bbox=dict(
                        beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                    loss_cls=dict(
                        loss_weight=1.0,
                        type='CrossEntropyLoss',
                        use_sigmoid=False),
                    num_classes=10,
                    reg_class_agnostic=True,
                    roi_feat_size=7,
                    type='Shared2FCBBoxHead'),
            ],
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=384,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            num_stages=3,
            stage_loss_weights=[
                1,
                0.5,
                0.25,
            ],
            type='CascadeRoIHead'),
        rpn_head=dict(
            anchor_generator=dict(
                ratios=[
                    0.5,
                    1.0,
                    2.0,
                ],
                scales=[
                    2,
                    4,
                    8,
                ],
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=384,
            in_channels=384,
            loss_bbox=dict(
                beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
            type='RPNHead'),
        test_cfg=dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.05),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        train_cfg=dict(
            rcnn=[
                dict(
                    assigner=dict(
                        ignore_iof_thr=-1,
                        match_low_quality=False,
                        min_pos_iou=0.5,
                        neg_iou_thr=0.5,
                        pos_iou_thr=0.5,
                        type='MaxIoUAssigner'),
                    debug=False,
                    pos_weight=-1,
                    sampler=dict(
                        add_gt_as_proposals=True,
                        neg_pos_ub=-1,
                        num=512,
                        pos_fraction=0.25,
                        type='RandomSampler')),
                dict(
                    assigner=dict(
                        ignore_iof_thr=-1,
                        match_low_quality=False,
                        min_pos_iou=0.6,
                        neg_iou_thr=0.6,
                        pos_iou_thr=0.6,
                        type='MaxIoUAssigner'),
                    debug=False,
                    pos_weight=-1,
                    sampler=dict(
                        add_gt_as_proposals=True,
                        neg_pos_ub=-1,
                        num=512,
                        pos_fraction=0.25,
                        type='RandomSampler')),
                dict(
                    assigner=dict(
                        ignore_iof_thr=-1,
                        match_low_quality=False,
                        min_pos_iou=0.7,
                        neg_iou_thr=0.7,
                        pos_iou_thr=0.7,
                        type='MaxIoUAssigner'),
                    debug=False,
                    pos_weight=-1,
                    sampler=dict(
                        add_gt_as_proposals=True,
                        neg_pos_ub=-1,
                        num=512,
                        pos_fraction=0.25,
                        type='RandomSampler')),
            ],
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=2000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=2000)),
        type='CascadeRCNN'),
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
optim_wrapper = dict(
    clip_grad=dict(max_norm=5, norm_type=2),
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.0005),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.067, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=7,
        gamma=0.7,
        milestones=[
            3,
            5,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root='/data/ephemeral/dataset/',
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scale=(
                                1333,
                                800,
                            ),
                            type='Resize'),
                        dict(
                            keep_ratio=True, scale=(
                                666,
                                400,
                            ), type='Resize'),
                    ],
                    [
                        dict(prob=1.0, type='RandomFlip'),
                        dict(prob=0.0, type='RandomFlip'),
                    ],
                    [
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    [
                        dict(
                            meta_keys=(
                                'img_id',
                                'img_path',
                                'ori_shape',
                                'img_shape',
                                'scale_factor',
                                'flip',
                                'flip_direction',
                            ),
                            type='PackDetInputs'),
                    ],
                ],
                type='TestTimeAug'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=7,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/data/ephemeral/dataset/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=7, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='train_new.json',
            backend_args=None,
            data_prefix=dict(img='./'),
            data_root='/data/ephemeral/dataset/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            metainfo=dict(
                classes=(
                    'General trash',
                    'Paper',
                    'Paper pack',
                    'Metal',
                    'Glass',
                    'Plastic',
                    'Styrofoam',
                    'Plastic bag',
                    'Battery',
                    'Clothing',
                )),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.1,
                        2.0,
                    ),
                    scale=(
                        800,
                        800,
                    ),
                    type='RandomResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        800,
                        800,
                    ),
                    crop_type='absolute_range',
                    recompute_bbox=True,
                    type='RandomCrop'),
                dict(min_gt_bbox_wh=(
                    0.01,
                    0.01,
                ), type='FilterAnnotations'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=3,
        type='RepeatDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            800,
            800,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            800,
            800,
        ),
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(min_gt_bbox_wh=(
        0.01,
        0.01,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1333,
                    800,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    666,
                    400,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='train.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root='/data/ephemeral/dataset/',
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/data/ephemeral/dataset/train.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/cascade-rcnn_convnext_large_fpn_8xb8-amp-lsj-5e_last'
