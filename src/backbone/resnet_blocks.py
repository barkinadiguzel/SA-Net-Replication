import torch
import torch.nn as nn
from layers.conv_layer import conv3x3
from layers.activation import get_activation

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_sa=False, G=32):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.relu = get_activation("relu")
        self.use_sa = use_sa

        if use_sa:
            from sa.sa_block_modular import SA_Block
            self.sa = SA_Block(out_channels, G=G)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.use_sa:
            out = self.sa(out)
        out += identity
        out = self.relu(out)
        return out
