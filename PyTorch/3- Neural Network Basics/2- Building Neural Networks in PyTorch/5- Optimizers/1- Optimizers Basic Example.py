import torch.optim as optim

# Example optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01)    # Stochastic Gradient Descent
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam Optimizer

# Update weights
optimizer.step()
