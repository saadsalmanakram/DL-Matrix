import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Example dataset
class MyDataset(Dataset):
    def __init__(self):
        # Initialize data
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        # Return data and label
        pass

dataset = MyDataset()

# Initialize DistributedSampler
sampler = DistributedSampler(dataset)

# Create DataLoader with DistributedSampler
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

for data, label in dataloader:
    # Process data and labels
    pass
