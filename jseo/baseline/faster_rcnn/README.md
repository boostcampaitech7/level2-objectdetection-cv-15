## 🐣 Faster R-CNN Lightning Code (baseline) 🐣

### 사용 방법
---

```bash
# lightning 라이브러리 설치
pip install lightning==2.1 torch==1.12.1
pip install tensorboard
```

### 텐서 보드 실행
---

```bash
# faster_rcnn 디렉토리 이동
cd faster_rcnn

# tensorboard 실행
tensorboard --logdir=lightning_logs/
```

### 라이트닝 모델 구성
---

#### 1. init

- model 정의 (torchvision의 resnet50 backbone faster-rcnn)
- roi_heads의 box_predictor 커스텀 (마지막 out_features=11)
- train_dataset, val_dataset init

#### 2. forward

- images와 targets가 들어오면 loss_dict 반환
- images만 들어오면 예측 결과 반환

#### 3. eval_forward

- faster r-cnn의 내부 구현상 eval 모드에서 loss_dict을 return하지 않음
- loss_dict을 함께 return 하도록 커스텀된 코드
- 내부 동작은 forward와 동일

#### 4. training_step

- 학습 스텝
- 배치 형태로 들어오면, images, targets, image_ids로 받은 뒤
- images와 targets를 forward 하여 loss_dict을 얻음
- 해당 loss 들을 로깅한 뒤, losses를 반환 (optimizer가 해당 정보로 가중치 조정)

#### 5. validation_step

- 검증 스텝
- 마찬가지로 배치 형태로 들어오면, images, targets, image_ids로 받은 뒤
- images와 targets를 forward 하는데
- eval mode에서 forward를 하면, loss_dict을 리턴하지 않아 로깅 불가
- 해서, 커스텀한 eval_forward를 사용함
- loss_dict을 얻으면, 해당 loss 들을 로깅한 뒤, losses 반환

#### 6. configure_optimizers

- optimizer, scheduler와 같은 유틸 함수들 정의 및 반환


#### 7. train & val dataloader

- 해당 함수들이 dataloader를 return하면, 학습 및 검증 시 사용됨

