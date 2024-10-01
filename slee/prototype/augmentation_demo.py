import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from utils.transform import get_common_transform, get_config_transform
from utils.dataset import CustomDataset

st.set_page_config(page_title='Augmentation DEMO', layout='wide')

st.title('🐣 Data Augmentation DEMO')
st.markdown('---')

# st.info('증강을 적용하고 싶은 이미지를 업로드 해주세요!')
# uploaded_file = st.file_uploader('이미지 업로드', type=['jpg', 'png', 'jpeg'], label_visibility="hidden")


@st.cache_data
def load_image():
    annotation = '../dataset/train.json' # annotation 경로
    data_dir = '../dataset' # data_dir 경로

    dataset = CustomDataset(annotation, data_dir, get_common_transform())

    dump_images = []

    for i in range(50):
        image, target, image_id = dataset[i]
        dump_images.append(
            {
                'image': image,
                'target': target,
                'image_id': image_id
            }
        )
    
    return dump_images

def draw_bbox(image_info):
    image = image_info['image']
    target = image_info['target']
    image_id = image_info['image_id']

    basic_color_palette = [
        {"name": "Red", "hex": "#FF0000"},
        {"name": "Green", "hex": "#00FF00"},
        {"name": "Blue", "hex": "#0000FF"},
        {"name": "Yellow", "hex": "#FFFF00"},
        {"name": "Cyan", "hex": "#00FFFF"},
        {"name": "Magenta", "hex": "#FF00FF"},
        {"name": "Black", "hex": "#000000"},
        {"name": "White", "hex": "#FFFFFF"},
        {"name": "Orange", "hex": "#FFA500"},
        {"name": "Purple", "hex": "#800080"}
    ]


    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    for bbox, label in zip(target['boxes'], target['labels']):
        xmin, ymin, xmax, ymax = bbox

        # 레이블이 bytes인 경우 문자열로 변환
        if isinstance(label, bytes):
            label = label.decode('utf-8')  # bytes를 str로 변환
        elif isinstance(label, torch.Tensor):  # label이 텐서인 경우 처리
            label = str(label.item())  # 텐서를 문자열로 변환

        # st.write(label)

        # draw bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline=basic_color_palette[int(label) % 10]['hex'], width=3)

        # add label text
        # draw.text((xmin, ymin - 20), label, fill='red')
    return image

def apply_augmentation(image_info, config):
    image = image_info['image']
    target = image_info['target']
    image_id = image_info['image_id']

    transform = get_config_transform(config)

    sample = {
        'image': image,
        'bboxes': target['boxes'],
        'labels': target['labels']
    }

    sample = transform(**sample)
    image = sample['image']
    target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

    return {'image': image, 'target': target, 'image_id': image_id}




images = load_image()
selected_number = st.selectbox("이미지를 선택하세요:", range(0, 50))

config = {}

st.sidebar.header('Adjustment')

if st.sidebar.checkbox('Brightness'):
    config['brightness'] = {'limit': st.sidebar.slider('Brightness', min_value=-1.0, max_value=1.0, value=(0.2), step=0.01)}

if st.sidebar.checkbox('Contrast'):
    config['contrast'] = {'limit': st.sidebar.slider('Contrast', min_value=-1.0, max_value=1.0, value=(0.2), step=0.01)}

if st.sidebar.checkbox('Horizontal Flip'):
    config['flip'] = True

if st.sidebar.checkbox('Vertical Flip'):
    config['vflip'] = True

if st.sidebar.checkbox('Rotate'):
    config['rotate'] = {'limit': st.sidebar.slider('Rotate', 0, 180, 0)}

if st.sidebar.checkbox('Random Crop'):
    config['crop_height'] = st.sidebar.slider('Crop Height', min_value=0, max_value=1024, value=1024//2)
    config['crop_width'] = st.sidebar.slider('Crop Width', min_value=0, max_value=1024, value=1024//2)

if st.sidebar.checkbox('Affine'):
    config['affine'] = {}
    config['affine']['angle'] = st.sidebar.slider('Angle', -180, 180, 0)
    config['affine']['translate_x'] = st.sidebar.slider("translate_x (%)", -1.0, 1.0, 0.0, step=0.01)
    config['affine']['translate_y'] = st.sidebar.slider("translate_y (%)", -1.0, 1.0, 0.0, step=0.01)
    config['affine']['scale'] = st.sidebar.slider("scale", 0.1, 3.0, 1.0, step=0.1)
    config['affine']['shear'] = st.sidebar.slider("shear angle (도)", -30, 30, 0)

if st.sidebar.checkbox('Elastic Transform'):
    config['elastic'] = {}
    config['elastic']['alpha'] = st.sidebar.slider("alpha", 0.0, 200.0, 30.0, step=0.1)
    config['elastic']['sigma'] = st.sidebar.slider("sigma", 0.0, 30.0, 4.0, step=0.1)
    config['elastic']['alpha_affine'] = st.sidebar.slider("alpha_affine", 0.0, 20.0, 4.0, step=0.1)

if st.sidebar.checkbox('Gaussian Blur'):
    config['blur'] = st.sidebar.slider('Blur (Kernel size)', 3, 15, 3, step=2)

if st.sidebar.checkbox('Random Gamma'):
    config['gamma'] = st.sidebar.slider('Gamma', min_value=50, max_value=150, value=100)

if st.sidebar.checkbox('Hue Saturation Value'):
    config['hue_shift'] = st.sidebar.slider('Hue Shift', min_value=-20, max_value=20, value=0)
    config['saturation_shift'] = st.sidebar.slider('Saturation Shift', min_value=-30, max_value=30, value=0)
    config['value_shift'] = st.sidebar.slider('Value Shift', min_value=-20, max_value=20, value=0)

if st.sidebar.checkbox('Random Snow'):
    config['snow'] = st.sidebar.slider('Snow Intensity', 0.0, 0.3, 0.1, step=0.05)

if st.sidebar.checkbox('Channel Shuffle'):
    config['channel_shuffle'] = True

if st.sidebar.checkbox('Motion Blur'):
    config['motion_blur'] = st.sidebar.slider('Kernel Size', 3, 33, 5, step=2)

if st.sidebar.checkbox('Cutout'):
    config['cutout'] = st.sidebar.slider('Cutout Size', 100, 500, 100, step=100)


col1, col2 = st.columns(2)

with col1:
    st.image(draw_bbox(images[selected_number]), width=500, caption=f'원본 이미지')

if st.button("적용하기"):
    augmented_image = None
    augmented_image = apply_augmentation(images[selected_number], config)
    with col2:
        st.image(draw_bbox(augmented_image), width=500, caption=f'증강 기법을 사용한 이미지')

st.write(config)