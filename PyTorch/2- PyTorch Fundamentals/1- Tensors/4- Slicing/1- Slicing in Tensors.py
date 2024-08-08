import torch

tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# Slice from start to end
slice = tensor[0:2, 1:3]

print(f"Sliced tensor: \n{tensor[0:2, 1:3]}")