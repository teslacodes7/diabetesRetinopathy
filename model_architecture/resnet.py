import torch.nn as nn
from torchvision import models

class ResNetDR(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=5, pretrained=True):
        super(ResNetDR, self).__init__()
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet model")

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
