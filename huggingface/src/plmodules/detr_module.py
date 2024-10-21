import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.models.detr import DetrResNet50, DetrResNet101
from src.models.deformable_detr import DeformableDetrResNet50
from src.models.conditional_detr import ConditionalDetrResNet50, ConditionalDetrResNet101

class DetrModuleBase(pl.LightningModule):
    def __init__(self, config, model_class):
        super().__init__()
        self.config = config
        self.model = model_class
        self.val_map_metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')
        self.train_map_metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
       
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict, outputs

    def training_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)
        batch_size = batch["pixel_values"].size(0)

        self.log("training_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), batch_size=batch_size)

        preds, target = self.preprocess_for_metrics(outputs, batch['labels'])

        self.train_map_metric.update(preds, target)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)
        batch_size = batch["pixel_values"].size(0)

        self.log("validation_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item(), batch_size=batch_size)

        preds, target = self.preprocess_for_metrics(outputs, batch['labels'])

        self.val_map_metric.update(preds, target)

        return loss

    def test_step(self, batch, batch_idx):
        return self(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.config.optimizer.lr_backbone,
            },
        ]

        optimizer_class = getattr(torch.optim, self.config.optimizer.name)
        optimizer = optimizer_class(param_dicts, **self.config.optimizer.params)

        if hasattr(self.config, "scheduler"):
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)
            scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
            return [optimizer], [scheduler]
        else:
            return optimizer
        
    def preprocess_for_metrics(self, outputs, targets):
        # prprocess data type to calculate mAP
        target = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        pred_boxes = outputs.pred_boxes
        logits = outputs.logits
        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)

        preds = self.filter_predictions(pred_boxes, scores, labels)

        for t in target:
            if 'class_labels' in t:
                t['labels'] = t.pop('class_labels')

        return preds, target
    
    def filter_predictions(self, pred_boxes, scores, labels):
        preds = []
        for i in range(len(pred_boxes)):
            boxes = pred_boxes[i]
            scores_i = scores[i]
            labels_i = labels[i]

            # # NMS 적용
            # keep = ops.nms(boxes, scores_i, iou_threshold=0.5)  # IoU 임계값 설정
            # boxes = boxes[keep]
            # scores_i = scores_i[keep]
            # labels_i = labels_i[keep]

            # 임계값 0.3 이상인 것만 추출
            threshold_mask = scores_i >= 0.2
            boxes = boxes[threshold_mask]
            scores_i = scores_i[threshold_mask]
            labels_i = labels_i[threshold_mask]

            # valid_mask = labels_i != 10  # 배경 클래스를 제외하기 위한 마스크
            # boxes = boxes[valid_mask]
            # scores_i = scores_i[valid_mask]
            # labels_i = labels_i[valid_mask]

            preds.append({
                "boxes": boxes,  # NMS 적용된 바운딩 박스
                "scores": scores_i,  # NMS 후 신뢰도 점수
                "labels": labels_i   # NMS 후 클래스 레이블
            })
        
        return preds

class DetrResNet50Module(DetrModuleBase):
    def __init__(self, config):
        super().__init__(config, DetrResNet50())

class DetrResNet101Module(DetrModuleBase):
    def __init__(self, config):
        super().__init__(config, DetrResNet101())

class ConditionalDetrResNet50Module(DetrModuleBase):
    def __init__(self, config):
        super().__init__(config, ConditionalDetrResNet50())

class ConditionalDetrResNet101Module(DetrModuleBase):
    def __init__(self, config):
        super().__init__(config, ConditionalDetrResNet101())

class DeformableDetrResNet50Module(DetrModuleBase):
    def __init__(self, config):
        super().__init__(config, DeformableDetrResNet50())