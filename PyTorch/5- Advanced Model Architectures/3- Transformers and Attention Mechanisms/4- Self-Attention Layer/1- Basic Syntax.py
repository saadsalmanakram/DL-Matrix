import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        # x shape: (sequence_length, batch_size, d_model)
        output, attn_weights = self.multihead_attention(x, x, x)
        return output, attn_weights

# Example usage
d_model = 512
nhead = 8
self_attention = SelfAttention(d_model, nhead)
x = torch.rand(10, 32, d_model)  # (sequence_length, batch_size, d_model)
output, attn_weights = self_attention(x)
