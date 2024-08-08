import torch
from torch.utils.data import DataLoader, Dataset

# Custom dataset example
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example data
data = list(range(10000))

# Creating dataset and dataloader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterating through dataloader
for batch in dataloader:
    # Process each batch
    pass
