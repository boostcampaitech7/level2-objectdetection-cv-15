import torchvision
import os
import numpy as np

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transform, processor, train=True):
        ann_file = os.path.join(img_folder, ann_file)
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor
        self.train = train
        self.transform = transform

    def __getitem__(self, idx):
        # 이미지와 타겟을 COCO 형식으로 불러오기
        img, target = super(CocoDetection, self).__getitem__(idx)

        # 이미지 ID와 타겟 설정
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # 바운딩 박스, 카테고리, areas 및 is_crowd 정보 추출
        bboxes = [ann['bbox'] for ann in target['annotations']]
        category_ids = [ann['category_id'] + 1 for ann in target['annotations']]  # category_id +1
        areas = [ann['area'] for ann in target['annotations']]  # 면적 정보
        is_crowds = [ann['iscrowd'] for ann in target['annotations']]  # 군중 정보

        if self.train:
            # albumentations 변환 적용
            transformed = self.transform(image=np.array(img), bboxes=bboxes, category_ids=category_ids)

            # 변환된 이미지와 바운딩 박스 업데이트
            img = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']

        # processor를 사용하여 추가 전처리 (DETR용)
        new_target = [{'bbox': bbox, 'category_id': cat_id, 'area': area, 'iscrowd': is_crowd} 
                      for bbox, cat_id, area, is_crowd in zip(bboxes, category_ids, areas, is_crowds)]
        coco_target = {'image_id': image_id, 'annotations': new_target}
        encoding = self.processor(images=img, annotations=coco_target, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze()  # 배치 차원 제거
        target = encoding["labels"][0]  # 배치 차원 제거

        return pixel_values, target
