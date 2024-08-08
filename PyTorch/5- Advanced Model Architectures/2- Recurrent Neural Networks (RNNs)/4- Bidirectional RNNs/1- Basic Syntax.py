import torch
import torch.nn as nn

# Define a bidirectional RNN
class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Hidden size doubled for bidirectional RNN
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
model = BidirectionalRNN(input_size, hidden_size, output_size)
