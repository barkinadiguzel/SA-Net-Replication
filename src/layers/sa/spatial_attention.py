import torch
import torch.nn as nn
from layers.activation import get_activation
from layers.normalization import get_normalization

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.norm = get_normalization("groupnorm", in_channels, num_groups)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = get_activation("sigmoid")
    
    def forward(self, x):
        x_norm = self.norm(x)     
        x_conv = self.conv(x_norm)
        x_att = self.sigmoid(x_conv)
        return x * x_att
