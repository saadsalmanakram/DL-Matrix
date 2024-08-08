import torch
import torch.nn as nn
import torch.optim as optim

# Example model
model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()

# Example optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
