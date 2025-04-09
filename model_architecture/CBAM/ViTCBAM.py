import torch
import torch.nn as nn
from timm import create_model
from CBAM import CBAM



class ViT_CBAM(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=5, pretrained=True):
        super(ViT_CBAM, self).__init__()
        self.vit = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.cbam = CBAM(in_channels=self.vit.embed_dim)
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        # Extract CLS token and apply CBAM
        cls_feat = x[:, 0].unsqueeze(-1).unsqueeze(-1)  # reshape to (B, C, 1, 1)
        cls_feat = self.cbam(cls_feat).squeeze(-1).squeeze(-1)  # back to (B, C)
        return self.classifier(cls_feat)
