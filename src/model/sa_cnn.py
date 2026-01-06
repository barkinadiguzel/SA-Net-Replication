import torch
import torch.nn as nn
from backbone.resnet_blocks import BasicBlock
from layers.activation import get_activation

class SA_ResNet(nn.Module):
    def __init__(self, num_classes=1000, use_sa_layers=[2,3,4], G=32):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = get_activation("relu")

        self.layer1 = self._make_layer(64, 2, use_sa=False, G=G)
        self.layer2 = self._make_layer(128, 2, use_sa=2 in use_sa_layers, G=G)
        self.layer3 = self._make_layer(256, 2, use_sa=3 in use_sa_layers, G=G)
        self.layer4 = self._make_layer(512, 2, use_sa=4 in use_sa_layers, G=G)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, use_sa, G):
        layers = []
        for _ in range(blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, use_sa=use_sa, G=G))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
