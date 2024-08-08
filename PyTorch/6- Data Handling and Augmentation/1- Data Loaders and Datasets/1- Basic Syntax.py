import torch
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example data
data = torch.randn(100, 3)  # 100 samples, 3 features
labels = torch.randint(0, 2, (100,))  # 100 labels for binary classification

# Create dataset and data loader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterating through data
for batch_data, batch_labels in dataloader:
    print(batch_data, batch_labels)
