import torch
import torch.nn as nn

# L1 Loss function
l1_loss = nn.L1Loss()

# Example usage
predictions = torch.tensor([2.5, 0.0, 2.1])
targets = torch.tensor([3.0, -0.5, 2.0])

loss = l1_loss(predictions, targets)
print(f"L1 Loss: {loss.item()}")
