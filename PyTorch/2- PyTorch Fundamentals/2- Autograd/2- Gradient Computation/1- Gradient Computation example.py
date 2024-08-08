import torch

# Define tensors with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Compute gradients
z.backward()

# Access gradients
print(x.grad)  # tensor([2.0, 4.0, 6.0])
