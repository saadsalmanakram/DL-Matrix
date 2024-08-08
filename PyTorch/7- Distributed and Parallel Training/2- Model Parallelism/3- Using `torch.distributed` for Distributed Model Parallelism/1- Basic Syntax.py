import torch
import torch.distributed as dist
import torch.nn as nn

# Initialize the distributed environment
dist.init_process_group(backend='nccl')

# Define a simple model with two layers
class DistributedModelParallelNN(nn.Module):
    def __init__(self):
        super(DistributedModelParallelNN, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Create the model and move it to the appropriate device
model = DistributedModelParallelNN().to('cuda')

# Wrap the model for distributed training
model = nn.parallel.DistributedDataParallel(model)

# Example input
input_tensor = torch.randn(128, 512).to('cuda')

# Forward pass
output = model(input_tensor)
