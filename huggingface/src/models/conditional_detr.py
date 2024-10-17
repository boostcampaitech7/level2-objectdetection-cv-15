import torch
from torch import nn
from transformers import ConditionalDetrForObjectDetection

class ConditionalDetrBase(nn.Module):
    def __init__(self, model_name):
        super(ConditionalDetrBase, self).__init__()

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
        
        self.model = ConditionalDetrForObjectDetection.from_pretrained(
            model_name,
            id2label=self.id2label,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

class ConditionalDetrResNet50(ConditionalDetrBase):
    def __init__(self):
        super(ConditionalDetrResNet50, self).__init__("microsoft/conditional-detr-resnet-50")

class ConditionalDetrResNet101(ConditionalDetrBase):
    def __init__(self):
        super(ConditionalDetrResNet101, self).__init__("Omnifact/conditional-detr-resnet-101-dc5")
