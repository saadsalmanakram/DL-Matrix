import torch
import torch.nn as nn

# Smooth L1 Loss function
smooth_l1_loss = nn.SmoothL1Loss()

# Example usage
predictions = torch.tensor([2.5, 0.0, 2.1])
targets = torch.tensor([3.0, -0.5, 2.0])

loss = smooth_l1_loss(predictions, targets)
print(f"Smooth L1 Loss: {loss.item()}")
