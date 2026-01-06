import torch
import torch.nn as nn
from sa.channel_attention import ChannelAttention
from sa.spatial_attention import SpatialAttention

def channel_shuffle(x, groups):
    B, C, H, W = x.size()
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x

class SA_Block(nn.Module):
    def __init__(self, in_channels, G=32):
        super().__init__()
        self.G = G
        assert in_channels % G == 0, "in_channels must be divisible by G"
        self.sub_channels = in_channels // G
        self.sub_ch_half = self.sub_channels // 2

        # Her alt-feature için ayrı attention branch’leri
        self.channel_att = nn.ModuleList([
            ChannelAttention(self.sub_ch_half) for _ in range(G)
        ])
        self.spatial_att = nn.ModuleList([
            SpatialAttention(self.sub_ch_half) for _ in range(G)
        ])

    def forward(self, x):
        x_groups = torch.chunk(x, self.G, dim=1)
        out_groups = []

        for i, xg in enumerate(x_groups):
            x_0, x_1 = torch.chunk(xg, 2, dim=1)

            # --- Channel Attention ---
            xn = self.channel_att[i](x_0)

            # --- Spatial Attention ---
            xs = self.spatial_att[i](x_1)

            # --- Concatenate ---
            out_groups.append(torch.cat([xn, xs], dim=1))

        out = torch.cat(out_groups, dim=1)
        out = channel_shuffle(out, self.G)
        return out
