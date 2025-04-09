import torch.nn as nn
import torch.nn.functional as F
from CBAM import ChannelAttention, SpatialAttention

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class CBAM_SE_Hybrid(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM_SE_Hybrid, self).__init__()
        self.se = SEBlock(in_channels, reduction=ratio)
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.se(x)
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
