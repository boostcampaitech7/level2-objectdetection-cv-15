# YOLO Object Detection Models

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


