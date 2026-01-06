import torch.nn as nn

def get_normalization(name, num_channels, num_groups=32):
    name = name.lower()
    if name == "groupnorm":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif name == "layernorm":
        return nn.LayerNorm([num_channels, 1, 1])
    else:
        raise ValueError(f"{name} normalization not implemented")
