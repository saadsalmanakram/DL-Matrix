import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Indexing single element
element = tensor[0, 1]

# Indexing a slice
slice = tensor[:, 1:]

print(f"Element at (0, 1): {tensor[0, 1]}")
print(f"Slice of tensor: \n{tensor[:, 1:]}")