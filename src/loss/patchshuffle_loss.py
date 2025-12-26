import torch
import torch.nn as nn

class PatchShuffleLoss(nn.Module):
    def __init__(self, shuffle_prob=0.1):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss()
        self.shuffle_prob = shuffle_prob  

    def forward(self, outputs, targets, outputs_shuffled=None):
        # Base loss
        loss = self.base_loss(outputs, targets)
        
        # Weighted shuffle loss
        if outputs_shuffled is not None:
            weight = self.shuffle_prob / (1 - self.shuffle_prob)
            loss += weight * self.base_loss(outputs_shuffled, targets)
        
        return loss
