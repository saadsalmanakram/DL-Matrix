import torch
import torch.nn as nn

# Cross-Entropy Loss function
cross_entropy_loss = nn.CrossEntropyLoss()

# Example usage
# Predictions are logits (raw scores) before softmax
predictions = torch.tensor([[2.5, 0.3, 0.2], [0.1, 2.1, 1.2]])
# Targets are class indices (e.g., 0, 1, 2)
targets = torch.tensor([0, 1])

loss = cross_entropy_loss(predictions, targets)
print(f"Cross-Entropy Loss: {loss.item()}")
