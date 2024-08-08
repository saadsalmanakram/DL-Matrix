import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super(SimpleAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# Example usage
d_model = 512
attention = SimpleAttention(d_model)
query = torch.rand(10, 32, d_model)  # (sequence_length, batch_size, d_model)
key = torch.rand(10, 32, d_model)
value = torch.rand(10, 32, d_model)
output, attn_weights = attention(query, key, value)
