import torch

# Define tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([[4], [5], [6]])

# Broadcasting addition
c = a + b
print(c)
# Output:
# tensor([[5, 6, 7],
#         [6, 7, 8],
#         [7, 8, 9]])
