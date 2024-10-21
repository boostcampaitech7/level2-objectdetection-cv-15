import argparse
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble inference tool')
    parser.add_argument('config', help='inference config file path')
    parser.add_argument(
        '--method',
        choices=['nms', 'soft_nms', 'non_maximum_weighted', 'weighted_boxes_fusion'],
        default='weighted_boxes_fusion',
        help='select ensemble method [nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    print(cfg)

    csv_paths = list(cfg.csv_paths)
    data_path = str(cfg.data_path)
    iou_thr = float(cfg.iou_thr)  # float으로 변환
    save_filename = str(cfg.save_filename)

    method = args.method
    csv_df = [pd.read_csv(file) for file in csv_paths]
    image_ids = csv_df[0]['image_id'].tolist()

    ann = os.path.join(data_path, 'test.json')
    coco = COCO(ann)

    prediction_strings = []
    file_names = []

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        
        for df in csv_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()
            
            if len(predict_list)==0 or len(predict_list)==1:
                continue
                
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []
            
            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info['width']
                box[1] = float(box[1]) / image_info['height']
                box[2] = float(box[2]) / image_info['width']
                box[3] = float(box[3]) / image_info['height']
                box_list.append(box)
                
            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))
        
        if len(boxes_list):
            # method를 globals에서 찾아 실행
            boxes, scores, labels = globals()[method](boxes_list, scores_list, labels_list, iou_thr=iou_thr)

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += f"{int(label)} {score:.6f} {box[0] * image_info['width']} {box[1] * image_info['height']} {box[2] * image_info['width']} {box[3] * image_info['height']} "
        
        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'{save_filename}.csv', index=False)

    print(submission.head())
    print('Inference is complete.')

if __name__ == '__main__':
    main()
