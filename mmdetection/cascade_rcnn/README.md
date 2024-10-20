## 🏆 Performance 

|Model|Backbone|Neck|Head|Learning Method|Leaderborad mAP50|
|---|:---:|:---:|:---:|:---:|:---:|
|Cascade-RCNN|ConvNeXt-large|FPN|CascadeRoIHead, RPNHead|LinearLR_40e|0.6025|
|Cascade-RCNN|ConvNeXt-xlarge|FPN|CascadeRoIHead, RPNHead|LinearLR_40e|0.6250|
|Cascade-RCNN|ConvNeXt-xlarge|FPN|CascadeRoIHead, RPNHead x512|LinearLR_1x|0.6315|

