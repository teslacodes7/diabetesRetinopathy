import torch
import torch.nn as nn
from torchvision import models

class InceptionDR(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, aux_logits=False):
        super(InceptionDR, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=aux_logits)
        
        # Change classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # If aux_logits is True, change the auxiliary head as well
        if aux_logits:
            aux_in_features = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    def forward(self, x):
        if self.model.aux_logits:
            out, aux_out = self.model(x)
            return out, aux_out
        else:
            return self.model(x)
