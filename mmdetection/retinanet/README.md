## 🏆 Performance 

|Model|Backbone|Neck|Head|Learning Method|Leaderborad mAP50|
|---|:---:|:---:|:---:|:---:|:---:|
|RetinaNet|ResNet-50|FPN|FreeAnchorRetinaHead|StepLR_1x|0.1771|
|RetinaNet|ResNet-50|FPN|PISARetinaHead|StepLR_1x|0.2482|
|RetinaNet|ResNet-50|FPG|RetinaHead|StepLR_50e|0.292|
|RetinaNet|ResNet-50|FPN|RetinaHead|LinearLR + MultiStepLR_90k|0.389|
|RetinaNet|ResNet-50|FPN|RetinaHead|mstrain_3x|0.4527|
|RetinaNet|ResNet-101|FPN|RetinaHead|mstrain_3x|0.4591|
|RetinaNet|ResNeXt-101|FPN|RetinaHead|mstrain_3x|0.4976|
|RetinaNet|ConvNeXt-tiny|FPN|RetinaHead|mstrain_3x|0.4978|
|RetinaNet|ConvNeXt-large|FPN|RetinaHead|mstrain_3x|0.6073|
