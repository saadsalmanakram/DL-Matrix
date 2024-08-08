import torch
import torchvision.transforms as transforms

# Example data
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Normalize using Min-Max scaling
normalize = transforms.Normalize(mean=[data.min(0)[0].tolist()], std=[data.max(0)[0].tolist()])
normalized_data = normalize(data)
print(normalized_data)
