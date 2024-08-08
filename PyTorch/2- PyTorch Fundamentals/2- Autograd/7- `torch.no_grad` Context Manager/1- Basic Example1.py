import torch

# Define tensors
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Perform operations without tracking gradients
with torch.no_grad():
    z = y.sum()

print(x.grad)  # tensor([0., 0., 0.]) (no gradients are tracked)
