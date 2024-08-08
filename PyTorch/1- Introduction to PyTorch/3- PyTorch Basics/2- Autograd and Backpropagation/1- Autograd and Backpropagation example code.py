import torch

# Create tensor with requires_grad=True to track operations
x = torch.tensor([2.0], requires_grad=True)

# Define a function
y = x**2 + 3*x + 2

# Perform backpropagation
y.backward()

# Print the gradient
print(x.grad)  # Output: tensor([7.])
