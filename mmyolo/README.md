# 🤗 YOLO 🤗

## YOLO model for Object Detection Recycling Trash

### 사용 방법
---

#### 레포지토리 클론
```bash
git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-15.git
```

#### mmyolo 폴더로 이동
```bash
cd mmyolo
```

#### config file 수정
---
- data_configs
    - raw_data_path 경로 수정

- 모델 선택
    - Yolov5s, Yolov5b, Yolov5l, Yolov5x
    - Yolox_s, Yolox_b, Yolox_l, Yolox_x...등
- train_configs
    - model_module 수정 (선택한 모델의 클래스명)
    - wandb 수정 (모델명)
- processor_configs
    - path 수정 (선택한 모델의 processor path (models 폴더 참조))

#### help 명령어 사용
```bash
train.py --help
```

#### 학습 시작
```bash
train.py --config={config_path} {--use_wandb}
```

### 프로젝트 구조
```
├─configs
│  ├─augmentation_configs
│  ├─data_configs
│  ├─processor_configs
│  └─train_configs
│      └─train
├─src
│  ├─callbacks
│  ├─data
│  │  ├─collate_fns
│  │  ├─custom_datamodules
│  │  ├─datasets
│  │  ├─processors
│  ├─models
│  ├─plmodules
│  └─utils
└─tests
```

## 🏆 Performance 

|Model|Backbone|Neck|Head|Learning Method|Leaderborad mAP50|
|----|:----:|:----:|:----:|:----:|:----:|
|Yolov5s|YOLOv5CSPDarknet|YOLOv5PAFPN|YOLOv5Head|LinearLR_300e|0.2438|
|Yolov5x|YOLOv5CSPDarknet|YOLOv5PAFPN|YOLOv5Head|LinearLR_500e|0.3299|
|Yolov5x(No Validation set Split)|YOLOv5CSPDarknet|YOLOv5PAFPN|YOLOv5Head|LinearLR_500e|0.4425|
|Yolov11x|???|???|???|LinearLR_300e|0.3715|
|Yolov10x|YOLOXCSPDarknet|YOLOXPAFPN|YOLOXHead|QuadraticWarmup_CosineAnnealing_Constant_300e||
|Yolov10x|Swin-L-IN22K|YOLOXPAFPN|YOLOXHead|QuadraticWarmup_CosineAnnealing_Constant_300e||
|Yolov10x|ConvnNeXT|YOLOXHead|QuadraticWarmup_CosineAnnealing_Constant_300e||


