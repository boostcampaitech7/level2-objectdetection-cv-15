import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO

wandb.init(project="ultralytics", job_type="training")
model = YOLO("../pretrained/yolo11s.pt")
add_wandb_callback(model, enable_model_checkpointing=True)

model.train(project="yolov11s", data="../data_cfg/yolov11.yaml", epochs=5, imgsz=640)
model.val()

wandb.finish()
