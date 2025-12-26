import torch

def get_feature_map_size(input_size, layers):
    H, W = input_size
    x = torch.zeros(1, 3, H, W)
    for layer in layers:
        x = layer(x)
    _, _, H_out, W_out = x.shape
    return H_out, W_out

def flatten_feature_map(x):
    return x.view(x.size(0), -1)
