import torch
import torch.nn as nn
from timm import create_model
from ..CBAM import CBAM

class EfficientNetCBAM(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=5, pretrained=True):
        super(EfficientNetCBAM, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.cbam = CBAM(self.model.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.num_features, num_classes)
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x
