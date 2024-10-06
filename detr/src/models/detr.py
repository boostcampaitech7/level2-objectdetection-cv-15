import torch
from torch import nn
from transformers import DetrForObjectDetection

class DetrBase(nn.Module):
    def __init__(self, config, model_name):
        super(DetrBase, self).__init__()

        self.id2label = {
            0: 'N/A', 
            1: 'General trash', 
            2: 'Paper',
            3: 'Paper pack',
            4: 'Metal', 
            5: 'Glass', 
            6: 'Plastic', 
            7: 'Styrofoam', 
            8: 'Plastic bag', 
            9: 'Battery', 
            10: 'Clothing'
        }

        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            id2label=self.id2label,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True,
            revision="no_timm"
        )

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

class DetrResNet50(DetrBase):
    def __init__(self, config):
        super(DetrResNet50, self).__init__(config, "facebook/detr-resnet-50")

class DetrResNet101(DetrBase):
    def __init__(self, config):
        super(DetrResNet101, self).__init__(config, "facebook/detr-resnet-101")
