import torch
from torch.utils.data import Dataset

class LargeDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def __len__(self):
        return 10000  # Example length

    def __getitem__(self, idx):
        if self.data is None:
            self.data = torch.load(self.file_path, map_location='cpu')  # Memory-mapped file

        return self.data[idx]

# Usage
large_dataset = LargeDataset('large_data.pt')
dataloader = DataLoader(large_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Process each batch
    pass
