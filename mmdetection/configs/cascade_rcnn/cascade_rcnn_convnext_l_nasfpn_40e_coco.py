_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_40e.py', '../_base_/default_runtime.py'
]

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.5,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='NASFPN',  # NAS-FPN으로 변경
        stack_times=7,  # NAS-FPN에서 피처 스택 반복 횟수
        in_channels=[192, 384, 768, 1536],  # ConvNeXt의 각 스테이지 출력 채널
        out_channels=256,  # NAS-FPN 출력 채널
        start_level=0,  # 시작 피처 레벨
        add_extra_convs='on_input',  # 추가 피처 맵 생성
        norm_cfg=dict(type='BN', requires_grad=True),  # Batch Normalization 설정
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=40)

