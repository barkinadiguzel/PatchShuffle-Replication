import torch.nn as nn

def get_normalization(num_features, norm_type="batch"):
    if norm_type.lower() == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type.lower() == "layer":
        return nn.LayerNorm([num_features, 1, 1])
    else:
        raise ValueError(f"Unsupported normalization: {norm_type}")
