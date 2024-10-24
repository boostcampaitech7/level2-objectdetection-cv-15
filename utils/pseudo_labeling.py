import numpy as np
import pandas as pd
import json
from pandas import json_normalize

# 기존의 json data (Labeled data) ######################################################################
labeled_json_data_path = "/data/ephemeral/dataset/train.json" # 경로수정 필요
with open(labeled_json_data_path) as f:
	labeled_data = json.load(f)

# width     height      file_name      license     flickr_url  coco_url    date_captured  id
df_images = json_normalize(labeled_data['images'])
# image_id  category_id     area    bbox    iscrowd     id
df_annotations = json_normalize(labeled_data['annotations'])

# 마지막 요소의 값들 가져오기
# [1024 1024 'train/4882.jpg' 0 None None '2020-12-23 16:20:30' 4882]
width, height, _, license, flickr_url, coco_url, date_captured, image_id_1 = df_images.tail(1).values[0]
# [4882 1 149633.22 list([145.4, 295.4, 420.2, 356.1]) 0 23143]
image_id_2, category_id, area, bbox, iscrowd, anno_id = df_annotations.tail(1).values[0]

# 예측을 통해 나온 데이터 (Unlabeled data) ######################################################################
submission_csv = '/data/ephemeral/mmdetection/tools/mAP=0.6315.csv' # 경로수정 필요
data = pd.read_csv(submission_csv, keep_default_na=False)
data = data.values.tolist()
# print(data.head(5)) # class, confidence, x1, y1, x2, y2 형태

unlabeled = dict()
unlabeled['images'] = []
unlabeled['annotations'] = []
confidence_threshold = 0.4

for predict, image in data:
    if predict is None or predict.strip() == '':  # 예측하지 못한 데이터는 pass
        continue

    count = 0  # 어노테이션 개수
    split_predict = predict.split(' ')
    anns_length = len(split_predict) // 6  # 어노테이션 개수 계산
    
    temp_image = None
    for i in range(anns_length):
        temp_annotation = dict()

        class_ = int(split_predict[i * 6])
        confidence = float(split_predict[(i * 6) + 1])
        Left = float(split_predict[(i * 6) + 2])
        Top = float(split_predict[(i * 6) + 3])
        Right = float(split_predict[(i * 6) + 4])
        Bottom = float(split_predict[(i * 6) + 5])
        Width = Right - Left
        Height = Bottom - Top
        Area = round(Width * Height, 2)

        if confidence_threshold is not None and confidence < confidence_threshold:
            continue

        # Annotation 추가
        anno_id += 1
        count += 1
        temp_annotation['image_id'] = image_id_2 + 1
        temp_annotation['category_id'] = class_
        temp_annotation['area'] = Area
        temp_annotation['bbox'] = [round(Left, 1), round(Top, 1), round(Width, 1), round(Height, 1)]
        temp_annotation['iscrowd'] = iscrowd  # 마지막 데이터 그대로 이용
        temp_annotation['id'] = anno_id
        unlabeled['annotations'].append(temp_annotation)  # 어노테이션 추가

    # 어노테이션이 있는 경우에만 이미지 추가
    if count > 0:
        image_id_2 += 1
        temp_image = dict(
            width=width,  # 마지막 데이터 그대로 이용
            height=height,
            file_name=image,
            license=license,  # 마지막 데이터 그대로 이용
            flickr_url=flickr_url,
            coco_url=coco_url,
            date_captured=date_captured,
            id=image_id_2
        )
        unlabeled['images'].append(temp_image)  # 이미지 추가


# Labeled Data + Unlabeled Data ################################################################################
labeled_data['images'] += unlabeled['images']
labeled_data['annotations'] += unlabeled['annotations']
with open("/data/ephemeral/dataset/train_new.json", "w") as new_file:
	json.dump(labeled_data, new_file)