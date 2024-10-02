import json
import os
import random
import shutil

def convert_coco_to_yolo(coco_annotation_file, images_dir, output_dir, val_split=0.2):
    with open(coco_annotation_file) as f:
        coco = json.load(f)

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_images_dir = os.path.join(output_dir, 'train/images')
    train_labels_dir = os.path.join(output_dir, 'train/labels')
    val_images_dir = os.path.join(output_dir, 'val/images')
    val_labels_dir = os.path.join(output_dir, 'val/labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Mapping from category_id to class number
    category_mapping = {category['id']: i for i, category in enumerate(coco['categories'])}

    # Collect image filenames for splitting
    all_images = {img['id']: img['file_name'] for img in coco['images']}
    image_ids = list(all_images.keys())
    random.shuffle(image_ids)

    # Split into train and validation sets
    split_index = int(len(image_ids) * (1 - val_split))
    train_ids = image_ids[:split_index]
    val_ids = image_ids[split_index:]

    # Process each image
    for img_id in coco['images']:
        img_file = img_id['file_name'][6:]
        img_path = os.path.join(images_dir, img_file)
         
        # Determine if it's a training or validation image
        if img_id["id"] in train_ids:
            output_img_path = os.path.join(train_images_dir, img_file)
            label_file_path = os.path.join(train_labels_dir, img_file.replace('.jpg', '.txt'))
        else:
            output_img_path = os.path.join(val_images_dir, img_file)
            label_file_path = os.path.join(val_labels_dir, img_file.replace('.jpg', '.txt'))

        # Copy image to output
        if os.path.exists(img_path):
            shutil.copy(img_path, output_img_path)

        with open(label_file_path, 'w') as label_file:
            for annotation in coco['annotations']:
                if annotation['image_id'] == img_id:
                    category_id = annotation['category_id']
                    if category_id in category_mapping:
                        class_id = category_mapping[category_id]
                        x, y, width, height = annotation['bbox']
                        # Convert to YOLO format: class x_center y_center width height
                        x_center = (x + width / 2) / img_id['width']
                        y_center = (y + height / 2) / img_id['height']
                        width /= img_id['width']
                        height /= img_id['height']

                        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Usage
coco_annotation_file = '/data/ephemeral/level2-objectdetection-cv-15/dataset/train.json'
images_dir = '/data/ephemeral/level2-objectdetection-cv-15/dataset/train/'
output_dir = '../yolo_dataset/'

convert_coco_to_yolo(coco_annotation_file, images_dir, output_dir, val_split=0.2)

