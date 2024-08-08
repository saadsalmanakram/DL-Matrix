import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Concatenate along a specific dimension
concatenated = torch.cat((tensor1, tensor2), dim=0)

print(f"Concatenated tensor: \n{concatenated}")