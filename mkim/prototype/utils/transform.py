import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_common_transform():
    return A.Compose([
        A.Resize(1024, 1024),
       # A.Flip(p=0.5),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_config_transform(config):

    transforms_list = []

    # brightness 값이 있을 경우에만 추가
    if 'brightness' in config and config['brightness'] is not None:
        transforms_list.append(A.RandomBrightness(limit=config['brightness']['limit'], p=1))

    # contrast 값이 있을 경우에만 추가
    if 'contrast' in config and config['contrast'] is not None:
        transforms_list.append(A.RandomContrast(limit=config['contrast']['limit'], p=1))

    # rotate 값이 있을 경우에만 추가
    if 'rotate' in config and config['rotate'] is not None:
        transforms_list.append(A.Rotate(limit=config['rotate']['limit'], p=1))

    # affine 값이 있을 경우에만 추가
    if 'affine' in config and config['affine'] is not None:
        transforms_list.append(A.Affine(
            rotate=config['affine']['angle'],
            translate_percent=(config['affine']['translate_x'], config['affine']['translate_y']),
            scale=config['affine']['scale'],
            shear=config['affine']['shear'],
            p=1
        ))

    if 'elastic' in config and config['elastic'] is not None:
        transforms_list.append(A.ElasticTransform(
            alpha=config['elastic']['alpha'],
            sigma=config['elastic']['sigma'],
            alpha_affine=config['elastic']['alpha_affine'],
            p=1
        ))

    if 'blur' in config and config['blur'] is not None:
        transforms_list.append(A.GaussianBlur(blur_limit=config['blur'], p=1))

    if 'flip' in config and config['flip']:
        transforms_list.append(A.HorizontalFlip(p=1))

    if 'vflip' in config and config['vflip']:
        transforms_list.append(A.VerticalFlip(p=1))

    if 'crop_height' in config and 'crop_width' in config:
        transforms_list.append(A.RandomCrop(height=config['crop_height'], width=config['crop_width'], p=1))
        
    if 'gamma' in config and config['gamma'] is not None:
        transforms_list.append(A.RandomGamma(gamma_limit=config['gamma'], p=1))

    if 'hue_shift' in config and 'saturation_shift' in config and 'value_shift' in config:
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=config['hue_shift'],
                                                    sat_shift_limit=config['saturation_shift'],
                                                    val_shift_limit=config['value_shift'],
                                                    p=1))
    
    if 'clahe_clip' in config:
        transforms_list.append(A.CLAHE(clip_limit=config['clahe_clip'], p=1))

    if 'snow' in config:
        transforms_list.append(A.RandomSnow(snow_point_lower=config['snow'], p=1))

    if 'motion_blur' in config:
        transforms_list.append(A.MotionBlur(blur_limit=config['motion_blur'], p=1))

    if 'cutout' in config:
        transforms_list.append(A.Cutout(num_holes=1, max_h_size=config['cutout'], max_w_size=config['cutout'], p=1))

    if 'channel_shuffle' in config:
        transforms_list.append(A.ChannelShuffle(p=1))

    # bbox_params는 그대로 유지
    transform = A.Compose(
        transforms_list,
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


    return transform