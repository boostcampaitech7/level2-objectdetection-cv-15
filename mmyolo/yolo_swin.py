_base_ = './yolox_s_fast_8xb8-300e_coco.py'

deepen_factor = _base_.deepen_factor
#widen_factor = _base_.widen_factor

# deepen_factor = 1
widen_factor = 0.5
channels = [384, 768, 1536]
checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmdet.SwinTransformer', # Using SwinTransformer from mmdet
        embed_dims=192,
        depths=[2, 2, 18, 2],
        #num_heads=[3, 6, 12, 24],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        #out_indices=(1, 2, 3),
        out_indices=(0, 1, 2),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOXPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of SwinTransformer-Tiny output are [192, 384, 768], which do not match the original yolov5-s neck and need to be changed.
        #out_channels=channels
        ),
    bbox_head=dict(
        type='YOLOXHead',
        head_module=dict(
            type='YOLOXHeadModule',
            # in_channels=channels, # input channels of head need to be changed accordingly
            # widen_factor=widen_factor
            ))
)