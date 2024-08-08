import torch
import torch.nn as nn

# Define a simple network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Dummy data
inputs = torch.randn(10, 10)
targets = torch.randn(10, 1)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)
print(f'Loss: {loss.item()}')
