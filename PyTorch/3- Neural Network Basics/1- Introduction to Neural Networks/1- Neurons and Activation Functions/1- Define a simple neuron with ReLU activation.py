import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neuron with ReLU activation
class SimpleNeuron(nn.Module):
    def __init__(self):
        super(SimpleNeuron, self).__init__()
        self.linear = nn.Linear(10, 1)  # Example: input of size 10, output of size 1

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)  # Apply ReLU activation
        return x

# Create an instance of the neuron
model = SimpleNeuron()
input_tensor = torch.randn(1, 10)  # Example input
output = model(input_tensor)
print(output)
