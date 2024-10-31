## 🏆 Performance 

|Model|Backbone|Neck|Head|Learning Method|Leaderborad mAP50|
|----|:----:|:----:|:----:|:----:|:----:|
|Cascade-RCNN|resnet50|FPN|CascadeRoIHead, RPNHead|LinearLR_1x|0.4419|
|Cascade-RCNN|ConvNext-small|FPN|CascadeRoIHead, RPNHead|LinearLR_40e|0.5761|
|Cascade-RCNN|ConvNeXt-large|FPN|CascadeRoIHead, RPNHead|LinearLR_40e|0.6025|
|Cascade-RCNN|ConvNeXt-xlarge|FPN|CascadeRoIHead, RPNHead|LinearLR_40e|0.6250|
|Cascade-RCNN|ConvNeXt-xlarge|FPN|CascadeRoIHead, RPNHead x512|LinearLR_1x|0.6315|
|Cascade-RCNN|ConvNeXt-xlarge|FPN|CascadeRoIHead, RPNHead x512|LinearLR_1x + pseudo labeled|0.6333|
|Cascade-RCNN|ConvNeXt-large|FPN|CascadeRoIHead, RPNHead x384|lsj-3times + pseudo labeled|0.6405|

