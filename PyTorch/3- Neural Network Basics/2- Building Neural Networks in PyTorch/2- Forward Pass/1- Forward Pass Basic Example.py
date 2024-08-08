import torch

# Define an example input
input_tensor = torch.randn(1, 10)  # Batch size of 1, 10 features

# Forward pass
output = model(input_tensor)
print(output)
