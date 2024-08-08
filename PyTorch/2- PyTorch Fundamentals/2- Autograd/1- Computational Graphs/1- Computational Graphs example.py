import torch

# Define tensors
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Print computational graph
print(x.grad_fn)  # None (x is a leaf node)
print(y.grad_fn)  # <PowBackward0>
print(z.grad_fn)  # <SumBackward0>
