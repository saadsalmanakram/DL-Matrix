import torch

# Define tensors
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Compute gradients with autograd.grad
grads = torch.autograd.grad(outputs=z, inputs=x)

# Print gradients
print(grads[0])  # tensor([2.0, 4.0, 6.0])
