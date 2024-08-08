import torch

# Create tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Add tensors
c = torch.add(a, b)
print(c)  # Output: tensor([5, 7, 9])
