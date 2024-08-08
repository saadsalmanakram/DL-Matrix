import torch

# Define model parameters
w = torch.tensor([1.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

# Example of gradient accumulation
for _ in range(3):
    loss = w ** 2
    loss.backward()
    print(w.grad)  # Accumulated gradients
    optimizer.step()
    optimizer.zero_grad()  # Zero out gradients for next iteration
