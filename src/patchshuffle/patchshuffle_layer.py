import torch
import torch.nn as nn

class PatchShuffle(nn.Module):
    def __init__(self, patch_size=(2,2), shuffle_prob=0.1):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        self.shuffle_prob = shuffle_prob

    def forward(self, x):
        if not self.training:
            return x
        B, C, H, W = x.shape
        out = x.clone()
        for b in range(B):
            if torch.rand(1).item() < self.shuffle_prob:
                for c in range(C):
                    out[b, c] = self.shuffle_feature_map(out[b, c])
        return out

    def shuffle_feature_map(self, fmap):
        H, W = fmap.shape
        ph, pw = self.patch_h, self.patch_w
        out = fmap.clone()
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                patch = fmap[i:i+ph, j:j+pw]
                flat = patch.flatten()
                perm = torch.randperm(flat.numel())
                out[i:i+ph, j:j+pw] = flat[perm].view_as(patch)
        return out
