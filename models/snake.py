import torch
import torch.nn as nn

class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()  # Proper initialization
        self.alpha = nn.Parameter(torch.ones(channels))  # learnable per-channel weight

    def forward(self, x):
        alpha = self.alpha.view(1, -1, 1, 1)
        return x + (1.0 / alpha) * torch.sin(x * alpha) ** 2
