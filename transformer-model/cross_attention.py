import torch
import math
from torch import nn

# Scaled dot-product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Masking: set scores to -inf where mask is True
        if mask is not None:
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        # Softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights