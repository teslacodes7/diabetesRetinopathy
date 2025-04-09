import torch
import torch.nn as nn
from torchvision import models
from ..CBAM import CBAM
import torch.nn.functional as F



class DenseNetCBAM(nn.Module):
    def __init__(self, model_name='densenet121', num_classes=5, pretrained=True):
        super(DenseNetCBAM, self).__init__()

        if model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=pretrained)
        else:
            raise ValueError("Unsupported DenseNet version")

        self.cbam = CBAM(self.model.features.norm5.num_features)  # Usually 1024 or 1664
        in_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.model.features(x)
        features = self.cbam(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.model.classifier(out)
        return out
