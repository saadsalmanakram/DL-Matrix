import torch
import torch.nn as nn

# BCE Loss function
bce_loss = nn.BCELoss()

# Example usage
# Predictions are probabilities after sigmoid
predictions = torch.tensor([0.8, 0.4, 0.6])
# Targets are binary labels (0 or 1)
targets = torch.tensor([1.0, 0.0, 1.0])

loss = bce_loss(predictions, targets)
print(f"BCE Loss: {loss.item()}")
