import torch
import torch.nn as nn
from timm import create_model

class EfficientNetDR(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=5, pretrained=True):
        super(EfficientNetDR, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
