import torch
import torch.nn as nn

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

# Create the model
model = SimpleNN()

# Wrap the model using DataParallel
model = nn.DataParallel(model)

# Transfer model to the first GPU
model = model.to('cuda:0')

# Example input
input_tensor = torch.randn(128, 512).to('cuda:0')

# Forward pass
output = model(input_tensor)
