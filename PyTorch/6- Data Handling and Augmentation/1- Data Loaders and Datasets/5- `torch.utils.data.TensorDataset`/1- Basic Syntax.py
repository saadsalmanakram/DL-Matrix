from torch.utils.data import TensorDataset, DataLoader

# Example tensors
data = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
targets = torch.tensor([0, 1, 0], dtype=torch.long)

# Create TensorDataset
tensor_dataset = TensorDataset(data, targets)

# DataLoader example
dataloader = DataLoader(tensor_dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    data, targets = batch
    print(data, targets)
