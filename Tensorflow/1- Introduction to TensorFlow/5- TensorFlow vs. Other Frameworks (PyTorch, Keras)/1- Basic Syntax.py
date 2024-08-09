# TensorFlow and Keras Integration Example
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dense(10, activation='softmax')
])
print(model.summary())

# PyTorch equivalent of a simple model (for comparison)
import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = SimpleModel()
print(model)
