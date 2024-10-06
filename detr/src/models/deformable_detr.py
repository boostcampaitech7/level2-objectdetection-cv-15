import torch
from torch import nn
from transformers import DeformableDetrForObjectDetection

class DeformableDetrBase(nn.Module):
    def __init__(self, config, model_name):
        super(DeformableDetrBase, self).__init__()

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

        self.model = DeformableDetrForObjectDetection.from_pretrained(
            model_name,
            id2label=self.id2label,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True,
            revision="no_timm"
        )

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

class DeformableDetrResNet50(DeformableDetrBase):
    def __init__(self, config):
        super(DeformableDetrResNet50, self).__init__(config, "SenseTime/deformable-detr")

