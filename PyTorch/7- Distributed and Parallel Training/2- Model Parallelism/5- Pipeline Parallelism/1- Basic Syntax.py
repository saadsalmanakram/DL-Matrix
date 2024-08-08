import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

# Define a simple model with two stages
class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

# Create the two stages of the model
stage1 = Stage1().to('cuda:0')
stage2 = Stage2().to('cuda:1')

# Combine them using Pipe to enable pipeline parallelism
model = Pipe(nn.Sequential(stage1, stage2), chunks=2)

# Example input
input_tensor = torch.randn(128, 512).to('cuda:0')

# Forward pass
output = model(input_tensor)
