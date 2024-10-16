auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
data_root = '/data/ephemeral/home/level2-objectdetection-cv-15/dataset/'
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
gpu_ids = [
    0,
]
img_scales = [
    (
        1333,
        800,
    ),
    (
        666,
        400,
    ),
    (
        2000,
        1200,
    ),
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
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
    backbone=dict(
        arch='large',
        drop_path_rate=0.5,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth',
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
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
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
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=10,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
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
        add_extra_convs='on_input',
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.95, decay_type='layer_wise', num_layers=6),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            9,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
seed = 2024
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(img='./'),
        data_root='/data/ephemeral/home/level2-objectdetection-cv-15/dataset/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
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
    num_workers=7,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/data/ephemeral/home/level2-objectdetection-cv-15/dataset/test.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
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
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='train.json',
            backend_args=None,
            data_prefix=dict(img='./'),
            data_root=
            '/data/ephemeral/home/level2-objectdetection-cv-15/dataset/',
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
                    scale=[
                        (
                            1333,
                            640,
                        ),
                        (
                            1333,
                            800,
                        ),
                    ],
                    type='RandomResize'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=3,
        type='RepeatDataset'),
    num_workers=7,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        scale=[
            (
                1333,
                640,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomResize'),
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
                dict(keep_ratio=True, scale=(
                    2000,
                    1200,
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
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/retinanet_convnext_large_fpn_mstrain_3x'