# 🤗 DETR 🤗

## DETR model for Object Detection Recycling Trash

### 사용 방법
---

#### 레포지토리 클론
```bash
git clone https://github.com/boostcampaitech7/level2-objectdetection-cv-15.git
```

#### detr 폴더로 이동
```bash
cd detr
```

#### config file 수정
---
- data_configs
    - raw_data_path 경로 수정

- 모델 선택
    - DetrResNet50, DetrResNet101 
    - ConditionalDetrResNet50, ConditionalDetrResNet101
    - DeformableDetrResNet50

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