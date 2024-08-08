import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create dataset, model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Dummy dataset
inputs = torch.randn(10, 10)
targets = torch.randn(10, 1)

# Training loop
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, targets)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
