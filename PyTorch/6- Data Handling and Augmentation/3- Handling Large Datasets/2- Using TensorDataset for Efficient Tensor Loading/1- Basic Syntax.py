import torch
from torch.utils.data import TensorDataset, DataLoader

# Example data
data = torch.randn(10000, 3, 32, 32)
labels = torch.randint(0, 10, (10000,))

# Create TensorDataset
tensor_dataset = TensorDataset(data, labels)

# Create DataLoader
dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterating through dataloader
for inputs, targets in dataloader:
    # Process each batch
    pass
