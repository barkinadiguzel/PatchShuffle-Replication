import torch.nn as nn

def get_pooling(pool_type="max", kernel_size=2, stride=2):
    if pool_type.lower() == "max":
        return nn.MaxPool2d(kernel_size, stride)
    elif pool_type.lower() == "avg":
        return nn.AvgPool2d(kernel_size, stride)
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")
