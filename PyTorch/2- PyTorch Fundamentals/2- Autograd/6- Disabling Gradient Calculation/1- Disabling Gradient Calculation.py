import torch

# Define tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Disable gradient computation
with torch.no_grad():
    y = x ** 2

print(x.grad)  # None (no gradient computed for this operation)
