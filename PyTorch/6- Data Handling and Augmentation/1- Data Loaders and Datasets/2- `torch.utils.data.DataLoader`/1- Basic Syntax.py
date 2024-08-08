import torch
from torch.utils.data import DataLoader

# Assume `dataset` is an instance of `torch.utils.data.Dataset`
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# Example usage
for batch in dataloader:
    data, targets = batch
    print(data, targets)
