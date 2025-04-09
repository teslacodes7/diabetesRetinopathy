import torch
import torch.nn as nn
from torchvision import models

class DenseNetDR(nn.Module):
    def __init__(self, model_name='densenet121', num_classes=5, pretrained=True):
        super(DenseNetDR, self).__init__()
        
        if model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=pretrained)
        elif model_name == 'densenet201':
            self.model = models.densenet201(pretrained=pretrained)
        else:
            raise ValueError("Unsupported DenseNet model")

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
