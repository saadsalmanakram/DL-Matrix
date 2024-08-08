import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(d_model, 1)  # Example output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

# Example usage
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048

model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
src = torch.rand(10, 32, d_model)  # (sequence_length, batch_size, d_model)
tgt = torch.rand(20, 32, d_model)
output = model(src, tgt)
