import torch
import torch.nn as nn
from torchvision import models
from CBAM import CBAM


class InceptionCBAM(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, aux_logits=False):
        super(InceptionCBAM, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=aux_logits)
        
        self.cbam = CBAM(in_channels=2048)  # Output feature size of InceptionV3
        self.model.fc = nn.Linear(2048, num_classes)

        if aux_logits:
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        if self.model.aux_logits:
            x, aux = self.model(x)
            x = self.cbam(x)
            return self.model.fc(x), aux
        else:
            x = self.model(x)
            x = self.cbam(x)
            return self.model.fc(x)
