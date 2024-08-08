import torch
import torch.nn as nn

# Example of creating a basic loss function
loss_function = nn.MSELoss()  # Mean Squared Error Loss

# Example usage
predictions = torch.tensor([0.5, 0.8, 0.3])
targets = torch.tensor([1.0, 0.0, 0.0])

loss = loss_function(predictions, targets)
print(f"Loss: {loss.item()}")
