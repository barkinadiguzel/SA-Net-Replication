import torch
import torch.nn as nn
from layers.activation import get_activation

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = get_activation("sigmoid")
    
    def forward(self, x):
        s = self.gap(x)      
        s = self.fc(s)       
        s = self.sigmoid(s)  
        return x * s
