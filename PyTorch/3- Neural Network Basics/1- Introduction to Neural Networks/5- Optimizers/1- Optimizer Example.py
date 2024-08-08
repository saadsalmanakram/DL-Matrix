import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Dummy data
inputs = torch.randn(10, 10)
targets = torch.randn(10, 1)

# Training step
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f'Loss after optimization: {loss.item()}')
