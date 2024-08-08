import torch
import torch.nn as nn

# MSE Loss function
mse_loss = nn.MSELoss()

# Example usage
predictions = torch.tensor([2.5, 0.0, 2.1])
targets = torch.tensor([3.0, -0.5, 2.0])

loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item()}")
