import torch
from torch.utils.data import Dataset

class CachedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            # Simulate expensive computation or IO
            item = self.data[idx] ** 2
            self.cache[idx] = item
            return item

# Example usage
data = list(range(10000))
cached_dataset = CachedDataset(data)
dataloader = DataLoader(cached_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Process each batch
    pass
