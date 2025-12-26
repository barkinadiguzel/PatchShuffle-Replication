import torch
import torch.nn as nn
from backbone.vgg_blocks import VGGBlock
from patchshuffle.patchshuffle_layer import PatchShuffle
from layers.pooling import get_pooling

class PatchShuffleVGG(nn.Module):
    def __init__(self, num_classes=10, shuffle_prob=0.1, patch_size=(2,2)):
        super().__init__()
        self.block1 = VGGBlock(3, 32)
        self.ps1 = PatchShuffle(patch_size, shuffle_prob)

        self.block2 = VGGBlock(32, 64)
        self.ps2 = PatchShuffle(patch_size, shuffle_prob)

        self.block3 = VGGBlock(64, 128)
        self.ps3 = PatchShuffle(patch_size, shuffle_prob)

        # Flatten + FC
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*4*4, num_classes)  # assuming input 32x32

    def forward(self, x):
        x = self.block1(x)
        x = self.ps1(x)
        x = self.block2(x)
        x = self.ps2(x)
        x = self.block3(x)
        x = self.ps3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
