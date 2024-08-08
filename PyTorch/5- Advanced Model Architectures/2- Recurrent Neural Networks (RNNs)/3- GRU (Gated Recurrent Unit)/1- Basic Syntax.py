import torch
import torch.nn as nn

# Define a GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
model = GRUModel(input_size, hidden_size, output_size)
