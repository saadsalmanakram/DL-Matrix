import torch
import torch.distributed as dist
import torch.nn as nn

# Initialize the distributed environment
dist.init_process_group(backend='nccl')

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

# Create the model and transfer it to the GPU
model = SimpleNN().to('cuda')

# Wrap the model using DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)

# Example input
input_tensor = torch.randn(128, 512).to('cuda')

# Forward pass
output = model(input_tensor)
