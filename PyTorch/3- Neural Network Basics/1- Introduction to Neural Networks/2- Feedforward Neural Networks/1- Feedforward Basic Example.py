import torch
import torch.nn as nn

# Define a feedforward neural network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # Input layer to hidden layer
        self.fc2 = nn.Linear(20, 1)   # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.fc2(x)
        return x

# Create an instance of the network
model = FeedforwardNN()
input_tensor = torch.randn(1, 10)  # Example input
output = model(input_tensor)
print(output)
