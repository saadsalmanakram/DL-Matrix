import torch
import torch.nn as nn

# Hinge Loss function
hinge_loss = nn.HingeEmbeddingLoss()

# Example usage
predictions = torch.tensor([1.0, -1.0, 0.5])
# Targets are -1 or 1
targets = torch.tensor([1, -1, 1])

loss = hinge_loss(predictions, targets)
print(f"Hinge Loss: {loss.item()}")
