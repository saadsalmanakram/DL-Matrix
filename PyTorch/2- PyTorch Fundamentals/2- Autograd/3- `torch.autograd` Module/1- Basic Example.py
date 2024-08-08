import torch

# Define a tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform operations
y = 2 * x
z = y.mean()

# Use autograd functions
z.backward()

# Access gradients
print(x.grad)  # tensor([0.6667, 0.6667, 0.6667])
