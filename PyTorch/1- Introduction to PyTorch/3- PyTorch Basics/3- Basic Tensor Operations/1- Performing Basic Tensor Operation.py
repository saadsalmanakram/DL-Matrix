import torch

# Create a 2x2 tensor
tensor = torch.tensor([[1, 2], [3, 4]])

# Reshape to a 1D tensor
reshaped_tensor = tensor.view(4)
print(reshaped_tensor)  # Output: tensor([1, 2, 3, 4])

# Access specific element
element = tensor[1, 0]
print(element)  # Output: tensor(3)

# Slice tensor
slice_tensor = tensor[:, 1]
print(slice_tensor)  # Output: tensor([2, 4])
