import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

# Example usage:
data = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
targets = torch.tensor([0, 1, 0], dtype=torch.long)
dataset = CustomDataset(data, targets)
