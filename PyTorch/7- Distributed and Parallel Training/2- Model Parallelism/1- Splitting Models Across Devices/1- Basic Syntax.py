import torch
import torch.nn as nn

# Define a simple model with two layers
class ModelParallelNN(nn.Module):
    def __init__(self):
        super(ModelParallelNN, self).__init__()
        # Place the first layer on the first GPU
        self.fc1 = nn.Linear(512, 512).to('cuda:0')
        # Place the second layer on the second GPU
        self.fc2 = nn.Linear(512, 512).to('cuda:1')

    def forward(self, x):
        # Transfer input to the first device
        x = x.to('cuda:0')
        # Forward pass through the first layer
        x = self.fc1(x)
        # Transfer the intermediate result to the second device
        x = x.to('cuda:1')
        # Forward pass through the second layer
        x = self.fc2(x)
        return x

# Create the model
model = ModelParallelNN()

# Example input
input_tensor = torch.randn(128, 512)  # Batch size 128, feature size 512

# Forward pass
output = model(input_tensor)
