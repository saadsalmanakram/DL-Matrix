import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_tensor = tensor.view(3, 2)

print(f"Reshaped tensor: \n{reshaped_tensor}")
