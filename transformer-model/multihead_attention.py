from torch import nn
import torch.nn.functional as F
from cross_attention import ScaledDotProductAttention

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        # q, k, v: [B, T, d_model]
        B, T_q, _ = q.size()

        # Project and reshape q, k, v to (B, T, d_model)
        q_proj = self.q_linear(q)
        k_proj = self.k_linear(k)
        v_proj = self.v_linear(v)

        # Reshape to (B, T, num_heads, d_k) and transpose to (B, num_heads, T, d_k)
        # This is done to allow for parallel computation across heads!! Really important
        q = q_proj.view(B, q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        k = k_proj.view(B, k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        v = v_proj.view(B, v.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        context, _ = self.attn(q, k, v, mask)

        # Combine heads and project back to d_model
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.d_k)

        output = self.out_linear(context)

        return output