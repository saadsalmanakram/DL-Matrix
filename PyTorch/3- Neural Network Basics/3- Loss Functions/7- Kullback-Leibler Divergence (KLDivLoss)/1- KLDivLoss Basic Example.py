import torch
import torch.nn as nn

# KL Divergence Loss function
kl_div_loss = nn.KLDivLoss(reduction='batchmean')

# Example usage
predictions = torch.log_softmax(torch.tensor([[0.4, 0.6], [0.5, 0.5]]), dim=1)
targets = torch.tensor([[0.5, 0.5], [0.4, 0.6]])

loss = kl_div_loss(predictions, targets)
print(f"KL Divergence Loss: {loss.item()}")
