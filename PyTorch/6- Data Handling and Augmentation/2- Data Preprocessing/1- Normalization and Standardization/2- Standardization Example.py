import torch
import torchvision.transforms as transforms

# Example data
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Standardize data
mean = data.mean(0)
std = data.std(0)
standardized_data = (data - mean) / std
print(standardized_data)
