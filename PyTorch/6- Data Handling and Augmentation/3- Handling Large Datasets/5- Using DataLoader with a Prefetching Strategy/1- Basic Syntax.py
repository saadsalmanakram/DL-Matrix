import torch
from torch.utils.data import DataLoader

# Example dataset and dataloader
class SimpleDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10000
    
    def __getitem__(self, index):
        return torch.randn(3, 32, 32), torch.tensor(index % 10)

dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Prefetching strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    # Process the batch
