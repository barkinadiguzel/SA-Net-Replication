import torch.nn as nn

def get_activation(name="relu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"{name} activation not implemented")
