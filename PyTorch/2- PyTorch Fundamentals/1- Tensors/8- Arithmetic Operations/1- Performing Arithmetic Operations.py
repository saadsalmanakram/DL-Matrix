import torch

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# Element-wise addition
sum_tensor = tensor1 + tensor2

# Element-wise multiplication
product_tensor = tensor1 * tensor2

print(f"Sum tensor: {sum_tensor}")
print(f"Product tensor: {product_tensor}")