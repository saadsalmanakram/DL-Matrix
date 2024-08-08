import torch

tensor = torch.tensor([[1, 2], [3, 4]])

# Get the shape
shape = tensor.shape

# Get the size
size = tensor.size()

# Get the data type
dtype = tensor.dtype

print(f"Shape: {tensor.shape}")
print(f"Size: {tensor.size()}")
print(f"Data Type: {tensor.dtype}")