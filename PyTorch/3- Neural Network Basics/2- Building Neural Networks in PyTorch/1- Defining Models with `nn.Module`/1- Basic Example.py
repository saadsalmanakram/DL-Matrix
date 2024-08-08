import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # First fully connected layer
        self.fc2 = nn.Linear(50, 1)   # Second fully connected layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)              # Apply second fully connected layer
        return x

model = MyModel()
