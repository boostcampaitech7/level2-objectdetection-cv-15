# RetinaNet Object Detection Models

## 🏆 Performance

| Model       | Backbone     | Neck                | Head              | Learning Method                | Leaderboard mAP50 |
|-------------|--------------|---------------------|-------------------|--------------------------------|-------------------|
| RetinaNet   | ResNet-50    | FPN                 | FreeAnchorRetinaHead | StepLR_1x                      | 0.1771            |
| RetinaNet   | ResNet-50    | FPN                 | PISARetinaHead    | StepLR_1x                      | 0.2482            |
| RetinaNet   | ResNet-50    | FPG                 | RetinaHead        | StepLR_50e                     | 0.292             |
| RetinaNet   | ResNet-50    | FPN                 | RetinaHead        | LinearLR + MultiStepLR_90k     | 0.389             |
| RetinaNet   | ResNet-50    | FPN                 | RetinaHead        | mstrain_3x                     | 0.4527            |
| RetinaNet   | ResNet-101   | FPN                 | RetinaHead        | mstrain_3x                     | 0.4591            |
| RetinaNet   | ResNeXt-101  | FPN                 | RetinaHead        | mstrain_3x                     | 0.4976            |
| RetinaNet   | ConvNeXt-tiny | FPN                | RetinaHead        | mstrain_3x                     | 0.4978            |
| RetinaNet   | ConvNeXt-large | FPN               | RetinaHead        | mstrain_3x                     | 0.6073            |



## 📝 retinanet_convnext_large_fpn_ms-640-800-3x_coco.py

The `retinanet_convnext_large_fpn_ms-640-800-3x_coco.py` configuration file is used to train a RetinaNet model with the ConvNeXt-large backbone and FPN neck for object detection on the COCO dataset. Below are some of the key settings and configurations used in this file:

- **Auto Scale LR**: `auto_scale_lr = dict(base_batch_size=16, enable=False)`
- **Checkpoint File**: Pretrained checkpoint used for ConvNeXt-large backbone from [OpenMMLab](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth)
- **Custom Imports**: Custom imports are included for `mmpretrain.models`.
- **Data Root**: The data root is set to `/data/ephemeral/home/level2-objectdetection-cv-15/dataset/`.
- **Dataset Type**: `dataset_type = 'CocoDataset'`
- **Default Hooks**:
  - **Checkpoint**: Saves model checkpoints every epoch with `max_keep_ckpts=3`.
  - **Logger**: Logs information every 50 iterations.
  - **Param Scheduler**: Uses `ParamSchedulerHook` to schedule parameter updates.
- **Model Architecture**:
  - **Backbone**: ConvNeXt-large, pretrained, with drop path rate of 0.5.
  - **Neck**: FPN with additional convolutions and output channels set to 256.
  - **Head**: RetinaHead with Focal Loss for classification and L1 Loss for bounding box regression.
  - **Test Configuration**: Uses Non-Maximum Suppression (NMS) with `iou_threshold=0.5`.
- **Optimizer**: AdamW optimizer with `learning rate = 0.0001`, `weight decay = 0.05`, and gradient clipping.
- **Learning Policy**:
  - **Linear Warmup**: Warmup for the first 500 iterations.
  - **MultiStepLR**: Steps at epoch 9 and 11, reducing learning rate by 0.1.
- **Training Configuration**: Total of 12 epochs (`max_epochs=12`), with validation interval after every epoch.

The model is configured for detection of 10 classes including: `General trash`, `Paper`, `Paper pack`, `Metal`, `Glass`, `Plastic`, `Styrofoam`, `Plastic bag`, `Battery`, and `Clothing`.

### ✅ Key Hyperparameters
- **Batch Size**: Training uses a batch size of 4.
- **Number of Workers**: 7 workers are used per GPU.
- **Image Scales**: The model is trained on multi-scale images, with sizes such as `(1333, 800)` and `(666, 400)`.
- **Optimizer Configuration**: `AdamW` optimizer is used with layer-wise learning rate decay.

### ✅ Model Usage
This model is suitable for applications that require accurate object detection on common waste items, such as recycling or environmental monitoring projects.

### ✅ Checkpoints and Logs
- **Checkpoints**: Checkpoints are saved every epoch to monitor training progress.
- **Logs**: Logs are recorded every 50 iterations for tracking metrics such as loss and learning rate.

For detailed hyperparameter settings and data processing pipelines, please refer to the full configuration file in the repository.

